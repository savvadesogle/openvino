// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05] Experimental GPU tensor-parallel support.
// Implements TPInferRequest: replicated-graph execution on k ranks, rank-0 outputs.
// See PLAN.md §4.4, §5 (S05).

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
}

void TPInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPInferRequest::infer");

    // Ranks are replicated: make sure each rank sees the same data as rank 0.
    replicate_inputs();

    // MVP: run ranks sequentially. Concurrent / device-side seam execution is S08+.
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

ov::SoPtr<ov::ITensor> TPInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    // Outputs and the "master" view of inputs both live on rank 0.
    return m_rank_requests.front()->get_tensor(port);
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
        // Outputs are served by rank 0 only.
        m_rank_requests.front()->set_tensor(port, tensor);
    }
}

std::vector<ov::SoPtr<ov::ITensor>> TPInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    return m_rank_requests.front()->get_tensors(port);
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
    // authoritative input/output views for this wrapper.
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
    // Ranks are identical in S05, so rank-0 states are representative.
    // Per-rank state aggregation (KV + linear-attention) is implemented in S10/S11.
    return m_rank_requests.front()->query_state();
}

}  // namespace ov::intel_gpu
