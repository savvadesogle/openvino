// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06/S07] Experimental GPU tensor-parallel support.
// TPShardTransform detects MatMul/FullyConnected candidates for tensor-parallel sharding
// and classifies each one (N-split / K-split / replica), recording the result in the model
// rt_info for the TPCoordinator to pick up.
//
// S06 ran as a pure analysis/identity pass (graph untouched).
// S07 makes it transform the graph for real: when it is constructed with a valid
// (rank, num_ranks) pair, it slices the CONSTANT weights of every N_SPLIT node along the
// output-channel (N) axis so that only this rank's slice is materialized/uploaded, and
// records the model outputs that must be all-gathered on rank 0.
// S08 extends the transform to K_SPLIT nodes: it slices the constant weights along the
// contraction (K) axis AND the activation operand along its last axis so each rank computes
// a partial GEMM sum, and records the model outputs that must be all-reduced (summed) on
// rank 0. Replicated bias/residual terms are recorded so the reducer adds them once.
// See PLAN.md §4.2, §4.3, §7.4.

#pragma once

#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Tensor-parallel seam detection + N-split weight slicing pass.
 *
 * Runs only when tensor parallelism is requested (the TransformationsPipeline gates it).
 * It always detects MatMul/FullyConnected nodes in deterministic topological order and
 * stores a TPSeam entry per detected node into the model rt_info under
 * ov::intel_gpu::tp_seams_rt_info_key().
 *
 * When constructed with @p num_ranks > 1 and a valid @p rank, the pass additionally
 * N-slices the constant weights (and bias) of N_SPLIT seams and K-slices the constant weights
 * plus the activation of K_SPLIT seams for that rank, recording the affected model outputs
 * (rt_info key: ov::intel_gpu::tp_output_seams_rt_info_key()) together with the replicated
 * additive terms of K-split outputs. In that case it returns true (graph modified); with the
 * default arguments it is an identity w.r.t. the graph (returns false) - the S06 behaviour.
 */
class TPShardTransform : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPShardTransform");

    /**
     * @param rank Index of the rank this model is being compiled for; < 0 disables slicing.
     * @param num_ranks Total number of tensor-parallel ranks; <= 1 disables slicing.
     */
    TPShardTransform(int32_t rank = -1, int32_t num_ranks = 1) : m_rank(rank), m_num_ranks(num_ranks) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    int32_t m_rank = -1;
    int32_t m_num_ranks = 1;
};

}  // namespace ov::intel_gpu
