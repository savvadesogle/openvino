// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06] Experimental GPU tensor-parallel support.
// TPShardTransform implementation: seam detection + conservative classification.
// Identity pass - sets model rt_info only, never changes graph structure.
// See PLAN.md §4.2, §4.3.

#include "tp_shard_transform.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov::intel_gpu {

namespace {

//! True for node kinds that are candidates for tensor-parallel sharding.
bool is_sharding_candidate(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v0::MatMul>(node) || ov::is_type<ov::intel_gpu::op::FullyConnected>(node);
}

//! Human-readable node type used in the seam record (FullyConnectedCompressed is a
//! subclass of FullyConnected, so it is checked first).
std::string classify_node_type(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<ov::intel_gpu::op::FullyConnectedCompressed>(node)) {
        return "FullyConnectedCompressed";
    }
    if (ov::is_type<ov::intel_gpu::op::FullyConnected>(node)) {
        return "FullyConnected";
    }
    if (ov::is_type<ov::op::v0::MatMul>(node)) {
        return "MatMul";
    }
    return node->get_type_name();
}

//! The weight matrix B is operand 1 for both MatMul and FullyConnected.
bool has_constant_weights(const std::shared_ptr<ov::Node>& node) {
    if (node->get_input_size() < 2) {
        return false;
    }
    return ov::is_type<ov::op::v0::Constant>(node->input_value(1).get_node());
}

//! Heuristic for "this GEMM is a projection whose partial N result is added back to the
//! residual stream" (O-proj / MLP down-proj in a transformer block). Such GEMMs are
//! K-split in PLAN §4.3 because the rank partials must be summed (all-reduce).
bool output_feeds_residual_add(const std::shared_ptr<ov::Node>& node) {
    for (const auto& target : node->output(0).get_target_inputs()) {
        if (ov::is_type<ov::op::v1::Add>(target.get_node())) {
            return true;
        }
    }
    return false;
}

}  // namespace

bool TPShardTransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<TPSeam> seams;

    // Deterministic order: topological (get_ordered_ops) and stable friendly names.
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_sharding_candidate(node)) {
            continue;
        }

        TPSeam seam;
        seam.node_name = node->get_friendly_name();
        seam.node_type = classify_node_type(node);

        // Conservative, deterministic classification (no slicing in S06):
        //  - no constant weights  -> REPLICA (cannot shard a non-constant operand safely);
        //  - output feeds an Add  -> K_SPLIT (residual projection, all-reduce(sum));
        //  - otherwise            -> N_SPLIT (entry GEMM, all-gather of output channels).
        if (!has_constant_weights(node)) {
            seam.kind = TPSeamKind::REPLICA;
            seam.collective = TPCollectiveKind::NONE;
        } else if (output_feeds_residual_add(node)) {
            seam.kind = TPSeamKind::K_SPLIT;
            seam.collective = TPCollectiveKind::ALL_REDUCE_SUM;
        } else {
            seam.kind = TPSeamKind::N_SPLIT;
            seam.collective = TPCollectiveKind::ALL_GATHER;
        }

        GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform seam: " << tp_seam_to_string(seam) << std::endl;
        seams.push_back(std::move(seam));
    }

    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform detected " << seams.size() << " candidate node(s)" << std::endl;

    // Transport the result to the TPCoordinator. The graph itself is left untouched,
    // hence the pass returns false (identity w.r.t. numerics/structure).
    set_tp_seams(model, std::move(seams));
    return false;
}

}  // namespace ov::intel_gpu
