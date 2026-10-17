// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06] Experimental GPU tensor-parallel support.
// TPShardTransform is an identity/analysis pass: it detects MatMul/FullyConnected
// candidates for tensor-parallel sharding and classifies each one (N-split / K-split /
// replica), then records the result in the model rt_info for the TPCoordinator to pick
// up. It NEVER modifies the graph, so the numeric result is unchanged (identity pass).
// Actual weight slicing and collectives are implemented in S07/S08.
// See PLAN.md §4.2, §4.3.

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Analysis-only tensor-parallel seam detection pass.
 *
 * Runs only when tensor parallelism is requested (the TransformationsPipeline gates it).
 * Detects MatMul/FullyConnected nodes in deterministic topological order and stores a
 * TPSeam entry per detected node into the model rt_info under
 * ov::intel_gpu::tp_seams_rt_info_key(). Returns false: the pass is an identity with
 * respect to the graph.
 */
class TPShardTransform : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPShardTransform");
    TPShardTransform() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
