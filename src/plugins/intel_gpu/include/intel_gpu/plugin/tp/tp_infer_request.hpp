// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05] Experimental GPU tensor-parallel support.
// TPInferRequest drives k per-rank inference requests (one per device) for a replicated
// graph: inputs are replicated to every rank, only rank-0 outputs are exposed.
// At this stage ranks are identical (no sharding yet), so no collective transport is needed.
// See PLAN.md §4.4, §5 (S05).

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::intel_gpu {

class TPCompiledModel;

/**
 * @brief Synchronous inference request for a tensor-parallel compiled model.
 *
 * Holds one sub-request per rank (built from the per-rank CompiledModel owned by
 * TPCompiledModel). Contract (mirrors CPU TP, see docs/notes/cpu-tp-pattern.md §6):
 *  - inputs   are replicated to every rank (same host tensor when the user provides one,
 *             otherwise the rank-0 input data is copied into each rank's own buffer);
 *  - outputs  are exposed from rank 0 only;
 *  - inference is executed sequentially rank-by-rank (MVP; device-side seams are S08+).
 *
 * Ranks are identical in S05, so no collective operation is required.
 */
class TPInferRequest : public ov::ISyncInferRequest {
public:
    using Ptr = std::shared_ptr<TPInferRequest>;

    explicit TPInferRequest(const std::shared_ptr<const TPCompiledModel>& compiled_model);
    ~TPInferRequest() override = default;

    void infer() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    void check_tensors() const override;

    // Ensures every rank sees the same input data as rank 0 before execution.
    void replicate_inputs();

    std::vector<ov::SoPtr<ov::ISyncInferRequest>> m_rank_requests;
};

}  // namespace ov::intel_gpu
