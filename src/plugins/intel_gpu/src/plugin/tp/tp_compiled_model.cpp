// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05] Experimental GPU tensor-parallel support.
// S16 adds export_model(): the per-rank cache envelope (one independent single-device blob per
// rank + TPPlan), imported back by Plugin::import_model_tensor_parallel.
// See PLAN.md §4.1, §4.4.

#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"

#include <sstream>
#include <string>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/plugin/tp/tp_infer_request.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"

namespace ov::intel_gpu {

TPCompiledModel::TPCompiledModel(const std::shared_ptr<const ov::IPlugin>& plugin,
                                 std::vector<std::shared_ptr<CompiledModel>> ranks,
                                 TPPlan plan)
    : ov::ICompiledModel(nullptr, plugin, ranks.front()->get_context_impl(), nullptr, nullptr),
      m_ranks(std::move(ranks)),
      m_plan(std::move(plan)) {
    OPENVINO_ASSERT(!m_ranks.empty(), "[GPU][TP] Tensor-parallel compiled model requires at least one rank");
    OPENVINO_ASSERT(m_ranks.size() == m_plan.ranks.size(),
                    "[GPU][TP] Rank models count (", m_ranks.size(), ") doesn't match plan ranks (", m_plan.ranks.size(), ")");

    // The base ICompiledModel was created without executors (each rank carries its own), but
    // the async wrapper needs a task executor to run the whole k-rank execution off the caller
    // thread. Build a dedicated executor so TP requests stay isolated from rank streams.
    set_task_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(
        ov::threading::IStreamsExecutor::Config{"Intel GPU TP plugin executor"}));

    // [S14] Rank scheduler: one stream per rank. TPInferRequest dispatches the k per-rank
    // sub-requests here (ITaskExecutor::run_and_wait) so they execute concurrently on their
    // own devices instead of serially. Kept at compiled-model scope: a pool, not per-infer
    // threads, because decode calls infer() once per token.
    m_rank_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
        ov::threading::IStreamsExecutor::Config{"Intel GPU TP rank scheduler",
                                                static_cast<int>(m_ranks.size())});

    build_output_ports();
}

// [S07] Ranks only materialize their N-slice of an N-split GEMM, so rank-0's native output
// shape is the slice shape. For assembly outputs advertise the FULL shape (all-gather
// equivalent); replicated outputs keep rank-0's native port as-is. S10 generalizes the split
// axis so head-sharded state outputs (split along an interior axis) also get a full-shape port.
void TPCompiledModel::build_output_ports() {
    const auto& rank0_outputs = m_ranks.front()->outputs();
    m_outputs.reserve(rank0_outputs.size());
    for (size_t i = 0; i < rank0_outputs.size(); ++i) {
        const auto& port = rank0_outputs[i];
        const auto* seam = m_plan.find_output_seam(i);
        // Only ALL_GATHER outputs need a synthetic full-shape port: their rank-0 native shape is
        // the slice. ALL_REDUCE_SUM (K-split) outputs already have the full shape on every rank,
        // so the native port is exposed as-is.
        if (seam == nullptr || seam->collective != TPCollectiveKind::ALL_GATHER || seam->shard_dim_size <= 0) {
            m_outputs.push_back(port);
            continue;
        }

        auto partial_shape = port.get_partial_shape();
        if (partial_shape.rank().is_static() && partial_shape.rank().get_length() >= 1) {
            const int64_t rank = partial_shape.rank().get_length();
            const int64_t axis_i = seam->split_axis < 0 ? rank + seam->split_axis : seam->split_axis;
            if (axis_i < 0 || axis_i >= rank) {
                m_outputs.push_back(port);
                continue;
            }
            const size_t axis = static_cast<size_t>(axis_i);
            if (partial_shape[axis].is_static() && partial_shape[axis].get_length() == seam->shard_dim_size) {
                // Already the full shape (e.g. graceful REPLICA fallback) - keep native port.
                m_outputs.push_back(port);
                continue;
            }
            if (partial_shape[axis].is_static()) {
                auto full = partial_shape;
                full[axis] = ov::Dimension(seam->shard_dim_size);
                auto param = std::make_shared<ov::op::v0::Parameter>(port.get_element_type(), full);
                param->set_friendly_name(port.get_node()->get_friendly_name() + "_tp_full");
                auto result = std::make_shared<ov::op::v0::Result>(param);
                result->set_friendly_name(port.get_node()->get_friendly_name());
                m_synthetic_output_nodes.push_back(param);
                m_synthetic_output_nodes.push_back(result);
                m_outputs.emplace_back(std::static_pointer_cast<const ov::Node>(result), 0);
                continue;
            }
        }
        m_outputs.push_back(port);
    }
}

const std::vector<ov::Output<const ov::Node>>& TPCompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& TPCompiledModel::inputs() const {
    return m_ranks.front()->inputs();
}

// S05: all k ranks execute the replicated graph; inputs are replicated to every rank,
// outputs are taken from rank 0 (see TPInferRequest).
std::shared_ptr<ov::IAsyncInferRequest> TPCompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPCompiledModel::create_infer_request");
    // The default ov::IAsyncInferRequest runs TPInferRequest::infer() as a single task.
    return create_async_infer_request<>();
}

std::shared_ptr<ov::ISyncInferRequest> TPCompiledModel::create_sync_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPCompiledModel::create_sync_infer_request");
    return std::make_shared<TPInferRequest>(std::static_pointer_cast<const TPCompiledModel>(shared_from_this()));
}

// [S16] Per-rank blob cache. A tensor-parallel compiled model is exported as a single envelope that
// carries one *independent* single-device blob per rank plus the TPPlan:
//
//     [ "TPSB" magic ][ uint32 version ][ uint64 num_ranks ]
//     [ TPPlan (ranks / device ids / seams / output seams / state seams) ]
//     [ per rank, in rank order: uint64 blob_size ; blob bytes ]
//
// Each rank's bytes come from its own CompiledModel::export_model(), i.e. the exact device-local
// program (sliced weights + per-rank ExecutionConfig). Ranks therefore never share a blob: on
// import each rank is deserialized with its own device engine/context (see
// Plugin::import_model_tensor_parallel). The blob is only ever written for TP groups built by
// compile_model_tensor_parallel, so the ordinary single-device cache format is untouched.
void TPCompiledModel::export_model(std::ostream& model) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPCompiledModel::export_model");
    OPENVINO_ASSERT(!m_ranks.empty(), "[GPU][TP] Compiled model not loaded");

    cldnn::BinaryOutputBuffer ob(model);
    ob << TENSOR_PARALLEL_BLOB_MAGIC;
    ob << TENSOR_PARALLEL_BLOB_VERSION;
    ob << static_cast<uint64_t>(m_ranks.size());

    save_tp_plan(ob, m_plan);

    for (const auto& rank : m_ranks) {
        std::stringstream rank_stream;
        rank->export_model(rank_stream);
        const std::string rank_blob = rank_stream.str();
        ob << static_cast<uint64_t>(rank_blob.size());
        ob << cldnn::make_data(rank_blob.data(), static_cast<uint64_t>(rank_blob.size()));
    }
    ob.flush();
}

std::shared_ptr<const ov::Model> TPCompiledModel::get_runtime_model() const {
    return m_ranks.front()->get_runtime_model();
}

ov::Any TPCompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        auto props = m_ranks.front()->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_ranks.name(), PropertyMutability::RO});
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_seams.name(), PropertyMutability::RO});
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_output_seams.name(), PropertyMutability::RO});
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_state_seams.name(), PropertyMutability::RO});
        props.emplace_back(ov::PropertyName{ov::hint::model_distribution_policy.name(), PropertyMutability::RO});
        return props;
    }
    if (name == ov::intel_gpu::tp_ranks) {
        return decltype(ov::intel_gpu::tp_ranks)::value_type{m_plan.ranks};
    }
    if (name == ov::intel_gpu::tp_seams) {
        std::vector<std::string> seams;
        seams.reserve(m_plan.seams.size());
        for (const auto& seam : m_plan.seams) {
            seams.push_back(tp_seam_to_string(seam));
        }
        return decltype(ov::intel_gpu::tp_seams)::value_type{std::move(seams)};
    }
    if (name == ov::intel_gpu::tp_output_seams) {
        std::vector<std::string> seams;
        seams.reserve(m_plan.output_seams.size());
        for (const auto& seam : m_plan.output_seams) {
            seams.push_back(tp_output_seam_to_string(seam));
        }
        return decltype(ov::intel_gpu::tp_output_seams)::value_type{std::move(seams)};
    }
    if (name == ov::intel_gpu::tp_state_seams) {
        std::vector<std::string> seams;
        seams.reserve(m_plan.state_seams.size());
        for (const auto& seam : m_plan.state_seams) {
            seams.push_back(tp_state_seam_to_string(seam));
        }
        return decltype(ov::intel_gpu::tp_state_seams)::value_type{std::move(seams)};
    }
    if (name == ov::hint::model_distribution_policy) {
        return decltype(ov::hint::model_distribution_policy)::value_type{ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    }
    // Delegate the rest to the rank-0 composed model (its config carries the same user properties).
    return m_ranks.front()->get_property(name);
}

void TPCompiledModel::set_property(const ov::AnyMap&) {
    OPENVINO_THROW_NOT_IMPLEMENTED("[GPU][TP] It's not possible to set property of an already compiled model. "
                                   "Set property to Core::compile_model during compilation");
}

}  // namespace ov::intel_gpu
