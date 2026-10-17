// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05] Experimental GPU tensor-parallel support.
// See PLAN.md §4.1, §4.4.

#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"

#include "intel_gpu/plugin/tp/tp_infer_request.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
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
}

const std::vector<ov::Output<const ov::Node>>& TPCompiledModel::outputs() const {
    return m_ranks.front()->outputs();
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

void TPCompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("[GPU][TP] Export of a tensor-parallel compiled model is not supported yet");
}

std::shared_ptr<const ov::Model> TPCompiledModel::get_runtime_model() const {
    return m_ranks.front()->get_runtime_model();
}

ov::Any TPCompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        auto props = m_ranks.front()->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_ranks.name(), PropertyMutability::RO});
        props.emplace_back(ov::PropertyName{ov::intel_gpu::tp_seams.name(), PropertyMutability::RO});
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
