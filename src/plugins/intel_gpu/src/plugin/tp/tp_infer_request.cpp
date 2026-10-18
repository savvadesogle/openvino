// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05/S07/S08/S10/S12/S14] Experimental GPU tensor-parallel support.
// TPInferRequest drives k per-rank inference requests (one per device).
//  - S05: replicated graph, inputs replicated to every rank, rank-0 outputs;
//  - S07: N-split ranks produce only their output-channel slice; rank-0 outputs marked as
//    all-gather are assembled from the per-rank slices (concatenation, no arithmetic);
//  - S08: K-split ranks produce a partial GEMM sum; rank-0 outputs marked as all-reduce are
//    summed in fp32 and the replicated bias/residual terms are added exactly once;
//  - S10: head-sharded state variables are aggregated from the per-rank slices on
//    query_state()/get_state() and scattered back on set_state();
//  - S12: inter-rank collectives (all-gather/all-reduce) are routed through the TPTransport
//    host-USM staging boundary; dynamic (partially-shaped) state variables are aggregated
//    runtime-shape-driven; inference is serialized per request to support async/multi-request use.
//  - S14: the k rank sub-requests are scheduled in parallel on the compiled model's k-stream rank
//    executor (one rank per stream/device). Input replication happens before the fan-out and
//    output/state assembly after every rank has finished, so the assembly order stays deterministic.
// See PLAN.md §4.3, §4.4, §7.4.

#include "intel_gpu/plugin/tp/tp_infer_request.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_transport.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace {

// Copies input data between two host tensors bound to the same port.
// Device (remote) tensors are rejected: replicated inputs are expected to be host-visible,
// which is the case for the default GPU I/O path (see docs/notes/gpu-plugin-arch.md §4).
void copy_host_tensor(const ov::SoPtr<ov::ITensor>& src,
                      const ov::SoPtr<ov::ITensor>& dst,
                      const ov::Output<const ov::Node>& port) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot replicate input tensor for port ", port);
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(src._ptr) == nullptr &&
                        std::dynamic_pointer_cast<ov::IRemoteTensor>(dst._ptr) == nullptr,
                    "[GPU][TP] Input replication for device (remote) tensors is not supported yet (S05)");

    if (port.get_element_type() == ov::element::string) {
        const auto count = dst->get_size();
        auto* s = src->data<std::string>();
        auto* d = dst->data<std::string>();
        for (size_t i = 0; i < count; ++i) {
            d[i] = s[i];
        }
        return;
    }

    const size_t bytes = std::min(src->get_byte_size(), dst->get_byte_size());
    std::memcpy(dst->data(), src->data(), bytes);
}

// [S10] Aggregates one logical state variable across the k ranks. For HEAD_SHARDED variables the
// per-rank states hold disjoint head-axis slices, so get_state() concatenates them into the full
// state and set_state() scatters a full state back into the per-rank slices. REPLICATED variables
// (the common case - dynamic / non-head-local chains) are rank-0 authoritative: get_state()
// returns rank 0 and reset()/set_state() fan out to every rank so ranks stay in sync.
class TPAggregatedState : public ov::IVariableState {
public:
    TPAggregatedState(const std::string& name,
                      std::vector<ov::SoPtr<ov::IVariableState>> ranks,
                      ov::intel_gpu::TPStateShardKind kind,
                      int64_t axis,
                      std::shared_ptr<ov::intel_gpu::TPTransport> transport)
        : ov::IVariableState(name),
          m_ranks(std::move(ranks)),
          m_kind(kind),
          m_axis(axis),
          m_transport(std::move(transport)) {}

    void reset() override {
        for (auto& s : m_ranks) {
            if (s) {
                s->reset();
            }
        }
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state, "[GPU][TP] Cannot set a null state tensor for ", m_name);
        if (m_kind != ov::intel_gpu::TPStateShardKind::HEAD_SHARDED || m_ranks.size() <= 1) {
            for (auto& s : m_ranks) {
                if (s) {
                    s->set_state(state);
                }
            }
            return;
        }
        // [S12] Derive the per-rank slice shape from the runtime (possibly dynamic batch/seq)
        // full state: only the head axis is static, the rest follows the incoming tensor.
        const size_t axis = static_cast<size_t>(m_axis);
        const auto full_shape = state->get_shape();
        OPENVINO_ASSERT(axis < full_shape.size(), "[GPU][TP] Invalid head axis for state ", m_name);
        OPENVINO_ASSERT(full_shape[axis] % m_ranks.size() == 0,
                        "[GPU][TP] State head axis ", full_shape[axis], " not divisible by ",
                        m_ranks.size(), " ranks for ", m_name);
        const size_t count = full_shape[axis] / m_ranks.size();
        std::vector<ov::SoPtr<ov::ITensor>> slices;
        slices.reserve(m_ranks.size());
        for (size_t r = 0; r < m_ranks.size(); ++r) {
            auto part_shape = full_shape;
            part_shape[axis] = count;
            slices.push_back(
                ov::SoPtr<ov::ITensor>(ov::make_tensor(state->get_element_type(), part_shape), nullptr));
        }
        m_transport->scatter_axis(state, axis, slices);
        for (size_t r = 0; r < m_ranks.size(); ++r) {
            if (m_ranks[r]) {
                m_ranks[r]->set_state(slices[r]);
            }
        }
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        if (m_kind != ov::intel_gpu::TPStateShardKind::HEAD_SHARDED || m_ranks.size() <= 1) {
            return m_ranks.front() ? m_ranks.front()->get_state() : ov::SoPtr<ov::ITensor>{};
        }
        auto first = m_ranks.front() ? m_ranks.front()->get_state() : ov::SoPtr<ov::ITensor>{};
        OPENVINO_ASSERT(first, "[GPU][TP] Rank 0 returned a null state for ", m_name);
        const size_t axis = static_cast<size_t>(m_axis);
        auto slice_shape = first->get_shape();
        OPENVINO_ASSERT(axis < slice_shape.size(), "[GPU][TP] Invalid head axis for state ", m_name);
        // [S12] Full shape follows the runtime rank slice; only the head axis is expanded. This
        // supports dynamic batch/sequence dims (Qwen `?,4,?,256` etc.).
        auto full_shape = slice_shape;
        full_shape[axis] = slice_shape[axis] * m_ranks.size();
        auto full = ov::SoPtr<ov::ITensor>(ov::make_tensor(first->get_element_type(), full_shape), nullptr);
        std::vector<ov::SoPtr<ov::ITensor>> slices;
        slices.reserve(m_ranks.size());
        for (size_t r = 0; r < m_ranks.size(); ++r) {
            slices.push_back(m_ranks[r] ? m_ranks[r]->get_state() : ov::SoPtr<ov::ITensor>{});
        }
        m_transport->all_gather_axis(slices, full, axis);
        return full;
    }

private:
    std::vector<ov::SoPtr<ov::IVariableState>> m_ranks;
    ov::intel_gpu::TPStateShardKind m_kind;
    int64_t m_axis;
    std::shared_ptr<ov::intel_gpu::TPTransport> m_transport;
};

}  // namespace

namespace ov::intel_gpu {

TPInferRequest::TPInferRequest(const std::shared_ptr<const TPCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model, "[GPU][TP] TPInferRequest requires a tensor-parallel compiled model");
    const auto& rank_models = compiled_model->get_rank_models();
    OPENVINO_ASSERT(!rank_models.empty(), "[GPU][TP] Tensor-parallel compiled model has no rank models");

    m_rank_requests.reserve(rank_models.size());
    for (const auto& rank_model : rank_models) {
        OPENVINO_ASSERT(rank_model, "[GPU][TP] Encountered a null rank model");
        m_rank_requests.push_back(rank_model->create_sync_infer_request());
    }
    // [S14] Rank scheduler: the shared k-stream executor owned by the compiled model. Each rank
    // runs on its own stream/device; the pool outlives the request (no threads per inference).
    m_rank_executor = compiled_model->get_rank_executor();

    // [S12] Inter-rank collectives run over the host-USM staging boundary (G0). One transport per
    // request owns the reusable staging buffer; it is shared with the state aggregators.
    m_transport = std::make_shared<TPTransport>(m_rank_requests.size());
    m_output_tensors.resize(get_outputs().size());
}

void TPInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPInferRequest::infer");

    // [S12] Serialize inference on a single request. Separate requests are independent objects
    // (each owns its rank sub-requests, transport and outputs), so concurrent async requests are
    // safe; a single request must not be re-entered concurrently.
    std::lock_guard<std::mutex> lock(m_infer_mutex);

    // Ranks receive replicated inputs: make sure each rank sees the same data as rank 0.
    // Done before the fan-out so every rank starts from the same inputs.
    replicate_inputs();

    // [S14] Run the k per-rank sub-requests in parallel (one per device); returns once all ranks
    // have completed. Collectives/assembly below run single-threaded afterwards (deterministic).
    schedule_ranks();

    // Gather the per-rank N-slices into the full outputs on rank 0.
    assemble_outputs();
}

void TPInferRequest::schedule_ranks() {
    // [S14] Dispatch one task per rank onto the shared k-stream executor and wait for all of them.
    // The rank sub-requests block inside infer() until their device work is done, so with k
    // streams the k tasks occupy k distinct worker threads and overlap across devices.
    if (m_rank_executor && m_rank_executor->get_streams_num() >= 1 && m_rank_requests.size() > 1) {
        std::vector<ov::threading::Task> tasks;
        tasks.reserve(m_rank_requests.size());
        for (auto& request : m_rank_requests) {
            tasks.emplace_back([&request] {
                request->infer();
            });
        }
        m_rank_executor->run_and_wait(tasks);
        return;
    }

    // Fallback (defensive): sequential execution preserving the deterministic rank order.
    for (auto& request : m_rank_requests) {
        request->infer();
    }
}

void TPInferRequest::replicate_inputs() {
    const auto& inputs = get_inputs();
    auto& master = m_rank_requests.front();
    for (const auto& port : inputs) {
        auto src = master->get_tensor(port);
        for (size_t r = 1; r < m_rank_requests.size(); ++r) {
            auto dst = m_rank_requests[r]->get_tensor(port);
            // When the user bound one shared tensor to every rank (default set_tensor path)
            // the pointers coincide and no copy is needed.
            if (src._ptr == dst._ptr) {
                continue;
            }
            copy_host_tensor(src, dst, port);
        }
    }
}

ov::SoPtr<ov::ITensor>& TPInferRequest::output_tensor(size_t index) const {
    std::lock_guard<std::mutex> lock(m_output_mutex);
    if (m_output_tensors.size() <= index) {
        m_output_tensors.resize(index + 1);
    }
    auto& tensor = m_output_tensors[index];
    if (!tensor) {
        const auto& port = get_outputs().at(index);
        tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(port.get_element_type(), port.get_shape()), nullptr);
    }
    return tensor;
}

ov::SoPtr<ov::ITensor>& TPInferRequest::ensure_output(size_t index,
                                                      const ov::Shape& shape,
                                                      const ov::element::Type& element_type) const {
    std::lock_guard<std::mutex> lock(m_output_mutex);
    if (m_output_tensors.size() <= index) {
        m_output_tensors.resize(index + 1);
    }
    auto& tensor = m_output_tensors[index];
    // (Re)allocate when the buffer is missing or the runtime shape/type changed (dynamic outputs).
    // A user-set tensor of the exact shape/type is preserved so the caller's buffer is filled.
    if (!tensor || tensor->get_shape() != shape || tensor->get_element_type() != element_type) {
        tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(element_type, shape), nullptr);
    }
    return tensor;
}

void TPInferRequest::assemble_outputs() {
    const auto compiled = std::static_pointer_cast<const TPCompiledModel>(get_compiled_model());
    const auto& plan = compiled->get_tp_plan();
    const size_t num_outputs = get_outputs().size();
    const size_t num_ranks = m_rank_requests.size();

    for (size_t i = 0; i < num_outputs; ++i) {
        const auto* seam = plan.find_output_seam(i);
        const bool sharded = seam != nullptr && num_ranks > 1;
        // [S12] Derive the assembled output shape from the *runtime* rank-0 output instead of the
        // possibly-dynamic declared port shape, so dynamic (batch/seq) outputs can be assembled.
        auto rank0_port = m_rank_requests.front()->get_outputs().at(i);
        auto rank0_tensor = m_rank_requests.front()->get_tensor(rank0_port);
        ov::Shape dst_shape = rank0_tensor->get_shape();
        size_t gather_axis = 0;
        const bool is_gather =
            sharded && seam->collective == TPCollectiveKind::ALL_GATHER && seam->shard_dim_size > 0;
        if (is_gather) {
            OPENVINO_ASSERT(!dst_shape.empty(), "[GPU][TP] Cannot assemble a scalar output");
            const int64_t full_rank = static_cast<int64_t>(dst_shape.size());
            const int64_t axis_i = seam->split_axis < 0 ? full_rank + seam->split_axis : seam->split_axis;
            OPENVINO_ASSERT(axis_i >= 0 && axis_i < full_rank, "[GPU][TP] Invalid all-gather axis ", seam->split_axis);
            gather_axis = static_cast<size_t>(axis_i);
            dst_shape[gather_axis] *= num_ranks;
        }

        auto& dst = ensure_output(i, dst_shape, rank0_tensor->get_element_type());
        if (sharded && seam->collective == TPCollectiveKind::ALL_REDUCE_SUM) {
            // [S08] K-split: sum the per-rank partial GEMM outputs (fp32, all-reduce sum).
            assemble_reduce_sum(i, *seam, dst);
        } else if (is_gather) {
            // All-gather: concatenate each rank's slice along the split axis (last axis for
            // N-split GEMM outputs, an interior head axis for head-sharded state outputs, S10).
            // [S12] Routed through the host-USM staging transport.
            std::vector<ov::SoPtr<ov::ITensor>> slices;
            slices.reserve(num_ranks);
            for (size_t r = 0; r < num_ranks; ++r) {
                auto rank_port = m_rank_requests[r]->get_outputs().at(i);
                slices.push_back(m_rank_requests[r]->get_tensor(rank_port));
            }
            m_transport->all_gather_axis(slices, dst, gather_axis);
        } else {
            // Replicated output: rank 0 is authoritative.
            copy_host_tensor(rank0_tensor, dst, get_outputs().at(i));
        }
    }
}

void TPInferRequest::assemble_reduce_sum(size_t output_index,
                                         const TPOutputSeam& seam,
                                         ov::SoPtr<ov::ITensor>& dst) {
    const size_t num_ranks = m_rank_requests.size();

    // Gather the per-rank partial GEMM outputs.
    std::vector<ov::SoPtr<ov::ITensor>> partials;
    partials.reserve(num_ranks);
    for (size_t r = 0; r < num_ranks; ++r) {
        auto rank_port = m_rank_requests[r]->get_outputs().at(output_index);
        partials.push_back(m_rank_requests[r]->get_tensor(rank_port));
    }

    // Resolve the replicated additive terms: constants carry their pre-expanded values, INPUT
    // terms (a residual stream) are read from rank 0's authoritative input tensor.
    std::vector<ov::SoPtr<ov::ITensor>> add_inputs;
    add_inputs.resize(seam.replicated_adds.size());
    for (size_t i = 0; i < seam.replicated_adds.size(); ++i) {
        const auto& add = seam.replicated_adds[i];
        if (add.kind == TPReplicatedAddKind::INPUT) {
            const auto& in_port = get_inputs().at(add.input_index);
            add_inputs[i] = m_rank_requests.front()->get_tensor(in_port);
        }
    }

    // [S12] fp32 all-reduce(sum) + replicated-term correction over the staging transport.
    m_transport->all_reduce_sum_fp32(partials, dst, seam.replicated_adds, add_inputs);
}

ov::SoPtr<ov::ITensor> TPInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);
    if (found_port.is_input()) {
        // The "master" view of inputs lives on rank 0.
        return m_rank_requests.front()->get_tensor(port);
    }
    // Outputs are served from the (possibly assembled) full-shape buffers owned here.
    return output_tensor(found_port.idx);
}

void TPInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);

    // Base-class bookkeeping + validation (element type / shape checks).
    ov::ISyncInferRequest::set_tensor(port, tensor);

    if (found_port.is_input()) {
        // Replicate the input on every rank. A host tensor is safe to share across devices
        // (each rank copies it into its own device memory during enqueue).
        for (auto& request : m_rank_requests) {
            request->set_tensor(port, tensor);
        }
    } else {
        // Outputs are owned by this request and filled during infer().
        std::lock_guard<std::mutex> lock(m_output_mutex);
        if (m_output_tensors.size() <= found_port.idx) {
            m_output_tensors.resize(found_port.idx + 1);
        }
        m_output_tensors[found_port.idx] = tensor;
    }
}

std::vector<ov::SoPtr<ov::ITensor>> TPInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);
    if (found_port.is_input()) {
        return m_rank_requests.front()->get_tensors(port);
    }
    return {get_tensor(port)};
}

void TPInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                 const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }
    // Batched (per-item) inputs are not part of the S05 scope.
    OPENVINO_THROW_NOT_IMPLEMENTED("[GPU][TP] Batched input tensors are not supported by the tensor-parallel request yet");
}

void TPInferRequest::check_tensors() const {
    // Tensors are owned by the per-rank requests and validated there; rank 0 holds the
    // authoritative input views and the assembled outputs are validated on set/alloc.
    return;
}

std::vector<ov::ProfilingInfo> TPInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t r = 0; r < m_rank_requests.size(); ++r) {
        auto rank_info = m_rank_requests[r]->get_profiling_info();
        for (auto& rec : rank_info) {
            rec.node_name = "rank" + std::to_string(r) + ": " + rec.node_name;
        }
        info.insert(info.end(), rank_info.begin(), rank_info.end());
    }
    return info;
}

std::vector<ov::SoPtr<ov::IVariableState>> TPInferRequest::query_state() const {
    // [S10] Aggregate the per-rank state variables. HEAD_SHARDED variables are concatenated along
    // their head axis; REPLICATED variables are rank-0 authoritative but reset/set_state fan out.
    const auto compiled = std::static_pointer_cast<const TPCompiledModel>(get_compiled_model());
    const auto& plan = compiled->get_tp_plan();
    const auto num_ranks = m_rank_requests.size();

    std::vector<std::vector<ov::SoPtr<ov::IVariableState>>> all_states;
    all_states.reserve(num_ranks);
    for (const auto& request : m_rank_requests) {
        all_states.push_back(request->query_state());
    }
    const auto& rank0_states = all_states.front();

    std::vector<ov::SoPtr<ov::IVariableState>> out;
    out.reserve(rank0_states.size());
    for (size_t i = 0; i < rank0_states.size(); ++i) {
        if (!rank0_states[i]) {
            continue;
        }
        const std::string name = rank0_states[i]->get_name();
        std::vector<ov::SoPtr<ov::IVariableState>> per_rank;
        per_rank.reserve(num_ranks);
        for (const auto& states : all_states) {
            ov::SoPtr<ov::IVariableState> found;
            for (const auto& s : states) {
                if (s && s->get_name() == name) {
                    found = s;
                    break;
                }
            }
            if (!found && states.size() == rank0_states.size()) {
                found = states[i];
            }
            per_rank.push_back(found);
        }
        if (per_rank.empty() || !per_rank.front()) {
            per_rank.assign(num_ranks, rank0_states[i]);
        }

        const auto* seam = plan.find_state_seam(name);
        if (seam != nullptr && seam->kind == TPStateShardKind::HEAD_SHARDED && num_ranks > 1) {
            // [S12] The full state shape is derived at runtime from the per-rank slices (dynamic
            // batch/sequence dims), so only the head axis is taken from the compile-time seam.
            out.emplace_back(std::make_shared<TPAggregatedState>(name,
                                                                 std::move(per_rank),
                                                                 TPStateShardKind::HEAD_SHARDED,
                                                                 seam->axis,
                                                                 m_transport),
                             nullptr);
        } else {
            out.emplace_back(std::make_shared<TPAggregatedState>(name,
                                                                 std::move(per_rank),
                                                                 TPStateShardKind::REPLICATED,
                                                                 -1,
                                                                 m_transport),
                             nullptr);
        }
    }
    return out;
}

}  // namespace ov::intel_gpu
