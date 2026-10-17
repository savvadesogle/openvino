// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05/S07] Experimental GPU tensor-parallel support.
// TPInferRequest drives k per-rank inference requests (one per device).
//  - S05: replicated graph, inputs replicated to every rank, rank-0 outputs;
//  - S07: N-split ranks produce only their output-channel slice; rank-0 outputs marked as
//    all-gather are assembled on the host from the per-rank slices (concatenation, no
//    arithmetic). K-split/all-reduce is S08.
// See PLAN.md §4.4, §5 (S05, S07).

#include "intel_gpu/plugin/tp/tp_infer_request.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"

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

// [S07] Copies the last-axis slice [offset, offset + count) of @p src into @p dst at the same
// position - the all-gather concatenation primitive (no arithmetic). @p src must be shaped
// like @p dst with the last dimension equal to @p count.
void copy_last_axis_region(const ov::SoPtr<ov::ITensor>& src,
                           const ov::SoPtr<ov::ITensor>& dst,
                           size_t offset,
                           size_t count) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot assemble output: null tensor");
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(src._ptr) == nullptr &&
                        std::dynamic_pointer_cast<ov::IRemoteTensor>(dst._ptr) == nullptr,
                    "[GPU][TP] Host assembly of device (remote) tensors is not supported yet (S07)");

    const auto full_shape = dst->get_shape();
    OPENVINO_ASSERT(!full_shape.empty(), "[GPU][TP] Cannot assemble a scalar output");
    const auto src_shape = src->get_shape();
    OPENVINO_ASSERT(src_shape.size() == full_shape.size() && src_shape.back() == count,
                    "[GPU][TP] Unexpected rank slice shape ", src_shape, " for full shape ", full_shape);
    OPENVINO_ASSERT(offset + count <= full_shape.back(), "[GPU][TP] All-gather region out of bounds");

    size_t outer = 1;
    for (size_t i = 0; i + 1 < full_shape.size(); ++i) {
        outer *= full_shape[i];
    }
    const size_t axis_dim = full_shape.back();

    const auto et = dst->get_element_type();
    if (et == ov::element::string) {
        auto* s = src->data<std::string>();
        auto* d = dst->data<std::string>();
        for (size_t o = 0; o < outer; ++o) {
            for (size_t j = 0; j < count; ++j) {
                d[o * axis_dim + offset + j] = s[o * count + j];
            }
        }
        return;
    }

    const size_t elem_bytes = et.size();
    const size_t row_bytes = count * elem_bytes;
    const auto* s = static_cast<const uint8_t*>(src->data());
    auto* d = static_cast<uint8_t*>(dst->data());
    for (size_t o = 0; o < outer; ++o) {
        std::memcpy(d + (o * axis_dim + offset) * elem_bytes, s + o * row_bytes, row_bytes);
    }
}

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
    m_output_tensors.resize(get_outputs().size());
}

void TPInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPInferRequest::infer");

    // Ranks receive replicated inputs: make sure each rank sees the same data as rank 0.
    replicate_inputs();

    // MVP: run ranks sequentially. Concurrent / device-side seam execution is S08+.
    for (auto& request : m_rank_requests) {
        request->infer();
    }

    // Gather the per-rank N-slices into the full outputs on rank 0.
    assemble_outputs();
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

void TPInferRequest::assemble_outputs() {
    const auto compiled = std::static_pointer_cast<const TPCompiledModel>(get_compiled_model());
    const auto& plan = compiled->get_tp_plan();
    const size_t num_outputs = get_outputs().size();
    const size_t num_ranks = m_rank_requests.size();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto& dst = output_tensor(i);
        const auto* seam = plan.find_output_seam(i);
        if (seam != nullptr && seam->shard_dim_size > 0 && seam->split_axis <= -1 && num_ranks > 1) {
            // All-gather: concatenate each rank's slice along the last (N) axis.
            const auto full_shape = dst->get_shape();
            const size_t count = full_shape.back() / num_ranks;
            for (size_t r = 0; r < num_ranks; ++r) {
                auto rank_port = m_rank_requests[r]->get_outputs().at(i);
                auto src = m_rank_requests[r]->get_tensor(rank_port);
                copy_last_axis_region(src, dst, r * count, count);
            }
        } else {
            // Replicated output: rank 0 is authoritative.
            auto rank_port = m_rank_requests.front()->get_outputs().at(i);
            auto src = m_rank_requests.front()->get_tensor(rank_port);
            copy_host_tensor(src, dst, get_outputs().at(i));
        }
    }
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
        output_tensor(found_port.idx) = tensor;
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
    // Per-rank state aggregation (KV + linear-attention) is implemented in S10/S11.
    return m_rank_requests.front()->query_state();
}

}  // namespace ov::intel_gpu
