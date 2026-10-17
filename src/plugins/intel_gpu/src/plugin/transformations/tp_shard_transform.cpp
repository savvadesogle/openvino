// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06/S07] Experimental GPU tensor-parallel support.
// TPShardTransform implementation:
//  - S06: seam detection + conservative classification (identity w.r.t. the graph);
//  - S07: real N-split - slice the constant weights (and bias) of N_SPLIT GEMMs along the
//    output-channel (N) axis for the current rank, so only this rank's slice is uploaded,
//    and record the model outputs that must be all-gathered on rank 0.
// K_SPLIT/REPLICA nodes are never touched (K-split + all-reduce is S08).
// See PLAN.md §4.2, §4.3.

#include "tp_shard_transform.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"

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

//! Purely element-wise ops through which an N-split slice may propagate unchanged. Used to
//! decide which model outputs can be assembled from per-rank slices (S07 all-gather).
bool is_elementwise(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v0::Abs>(node) || ov::is_type<ov::op::v0::Clamp>(node) ||
           ov::is_type<ov::op::v0::Elu>(node) || ov::is_type<ov::op::v0::Erf>(node) ||
           ov::is_type<ov::op::v0::Exp>(node) || ov::is_type<ov::op::v0::Log>(node) ||
           ov::is_type<ov::op::v0::Negative>(node) || ov::is_type<ov::op::v0::Relu>(node) ||
           ov::is_type<ov::op::v0::Sigmoid>(node) || ov::is_type<ov::op::v0::Softplus>(node) ||
           ov::is_type<ov::op::v0::Tanh>(node) || ov::is_type<ov::op::v1::Add>(node) ||
           ov::is_type<ov::op::v1::Divide>(node) || ov::is_type<ov::op::v1::Multiply>(node) ||
           ov::is_type<ov::op::v1::Subtract>(node) || ov::is_type<ov::op::v4::Swish>(node) ||
           ov::is_type<ov::op::v7::Gelu>(node);
}

//! Returns the transpose_b attribute for the GEMM kinds we know about.
bool get_transpose_b(const std::shared_ptr<ov::Node>& node, bool& transpose_b) {
    if (auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
        transpose_b = mm->get_transpose_b();
        return true;
    }
    if (auto fc = ov::as_type_ptr<ov::intel_gpu::op::FullyConnected>(node)) {
        transpose_b = fc->get_transpose_b();
        return true;
    }
    return false;
}

//! Axis of the weights tensor that carries the output channels (N). Uses the GEMM transpose
//! attribute when available, and falls back to matching the output's last dimension.
int64_t find_weight_split_axis(const std::shared_ptr<ov::Node>& node,
                               const ov::Shape& weights_shape,
                               int64_t n_out) {
    bool transpose_b = false;
    if (get_transpose_b(node, transpose_b) && !weights_shape.empty()) {
        const int64_t axis = transpose_b ? 0 : static_cast<int64_t>(weights_shape.size()) - 1;
        if (weights_shape[axis] == static_cast<size_t>(n_out)) {
            return axis;
        }
    }
    if (!weights_shape.empty() && static_cast<int64_t>(weights_shape.back()) == n_out) {
        return static_cast<int64_t>(weights_shape.size()) - 1;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(weights_shape.size()); ++i) {
        if (static_cast<int64_t>(weights_shape[i]) == n_out) {
            return i;
        }
    }
    return -1;
}

//! Slices a constant along @p axis into [offset, offset + count) (element-aligned types only).
std::shared_ptr<ov::op::v0::Constant> slice_constant(const std::shared_ptr<ov::op::v0::Constant>& src,
                                                     int64_t axis,
                                                     int64_t offset,
                                                     int64_t count) {
    const auto et = src->get_element_type();
    const auto shape = src->get_shape();
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()),
                    "[GPU][TP] N-split: invalid weights axis ", axis);
    // Sub-byte types (u4/i4) are handled in S09 (group quantization); reject explicitly here.
    OPENVINO_ASSERT(et.bitwidth() >= 8, "[GPU][TP] N-split slicing of sub-byte weight type ", et,
                    " is not supported yet (int4 is S09)");
    const size_t elem_bytes = et.size();
    const size_t axis_dim = shape[axis];
    OPENVINO_ASSERT(offset >= 0 && count >= 0 && static_cast<size_t>(offset + count) <= axis_dim,
                    "[GPU][TP] N-split: range [", offset, ", ", offset + count, ") out of bounds for axis dim ",
                    axis_dim);

    size_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    size_t inner = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }

    ov::Shape out_shape(shape.begin(), shape.end());
    out_shape[axis] = static_cast<size_t>(count);

    // Raw pointer walks the source as [outer, axis_dim, inner] blocks; each rank gets a
    // contiguous [count, inner] slab per outer index. Byte copy keeps the exact bit pattern.
    const auto* in = static_cast<const uint8_t*>(src->get_data_ptr());
    std::vector<uint8_t> out(outer * static_cast<size_t>(count) * inner * elem_bytes);
    const size_t slab_bytes = static_cast<size_t>(count) * inner * elem_bytes;
    for (size_t o = 0; o < outer; ++o) {
        const uint8_t* src_base = in + (o * axis_dim + static_cast<size_t>(offset)) * inner * elem_bytes;
        uint8_t* dst_base = out.data() + o * slab_bytes;
        std::memcpy(dst_base, src_base, slab_bytes);
    }
    return std::make_shared<ov::op::v0::Constant>(et, out_shape, out.data());
}

//! Slices an optional bias (operand 2) constant along its output-channel axis (0 for [N]).
void slice_bias_if_present(const std::shared_ptr<ov::Node>& node, int64_t n_out, int64_t offset, int64_t count) {
    if (node->get_input_size() < 3) {
        return;
    }
    auto bias = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr());
    if (!bias) {
        return;
    }
    const auto bias_shape = bias->get_shape();
    if (bias_shape.empty() || bias_shape[0] != static_cast<size_t>(n_out)) {
        return;
    }
    auto sliced = slice_constant(bias, 0, offset, count);
    sliced->set_friendly_name(bias->get_friendly_name() + "_tp_rank");
    ov::copy_runtime_info(bias, sliced);
    node->input(2).replace_source_output(sliced);
}

//! Reproduces the S06 classification for a candidate node.
void classify_seam(const std::shared_ptr<ov::Node>& node, TPSeam& seam) {
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
}

//! True if every consumer of @p node's output is element-wise or a Result, i.e. the sliced
//! output can safely be assembled from per-rank slices (S07 scope). Any other consumer (e.g.
//! a chained GEMM that needs the full N) is out of scope for S07.
bool consumers_are_slice_safe(const std::shared_ptr<ov::Node>& node) {
    for (const auto& target : node->output(0).get_target_inputs()) {
        const auto consumer = target.get_node()->shared_from_this();
        if (!is_elementwise(consumer) && !ov::is_type<ov::op::v0::Result>(consumer)) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool TPShardTransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const bool sharding_active = m_num_ranks > 1 && m_rank >= 0 && m_rank < m_num_ranks;
    const bool group_size_is_one = true;  // int4 group quantization is handled in S09

    std::vector<TPSeam> seams;
    std::vector<TPOutputSeam> output_seams;
    // Nodes whose output was actually sliced for this rank (name -> split geometry).
    std::unordered_set<const ov::Node*> sliced_nodes;
    // split geometry per sliced node (N and axis), keyed by node pointer.
    std::vector<std::pair<std::shared_ptr<ov::Node>, std::pair<int64_t, int64_t>>> sliced_info;

    bool graph_modified = false;

    // Deterministic order: topological (get_ordered_ops) and stable friendly names.
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_sharding_candidate(node)) {
            continue;
        }

        TPSeam seam;
        seam.node_name = node->get_friendly_name();
        seam.node_type = classify_node_type(node);
        classify_seam(node, seam);

        if (sharding_active && seam.kind == TPSeamKind::N_SPLIT) {
            const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
            const auto out_pshape = node->get_output_partial_shape(0);
            if (weights && out_pshape.rank().is_static() && out_pshape.rank().get_length() >= 1 &&
                out_pshape[out_pshape.rank().get_length() - 1].is_static()) {
                const int64_t n_out = out_pshape[out_pshape.rank().get_length() - 1].get_length();
                const int64_t axis = find_weight_split_axis(node, weights->get_shape(), n_out);
                const bool divisible = (n_out > 0) && (n_out % m_num_ranks == 0);
                const bool safe_consumers = consumers_are_slice_safe(node);
                const bool splittable = axis >= 0 && divisible && safe_consumers && group_size_is_one &&
                                        weights->get_shape()[axis] == static_cast<size_t>(n_out);

                if (splittable) {
                    const int64_t count = n_out / m_num_ranks;
                    const int64_t offset = static_cast<int64_t>(m_rank) * count;
                    auto sliced_weights = slice_constant(weights, axis, offset, count);
                    sliced_weights->set_friendly_name(weights->get_friendly_name() + "_tp_rank");
                    ov::copy_runtime_info(weights, sliced_weights);
                    node->input(1).replace_source_output(sliced_weights);
                    slice_bias_if_present(node, n_out, offset, count);

                    seam.split_axis = -1;  // N is the last dimension of the GEMM output
                    seam.shard_dim_size = n_out;

                    sliced_nodes.insert(node.get());
                    sliced_info.emplace_back(node, std::make_pair(seam.split_axis, seam.shard_dim_size));
                    graph_modified = true;
                    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform N-split rank " << m_rank << "/" << m_num_ranks
                                   << " " << node->get_friendly_name() << ": N=" << n_out
                                   << " slice=[" << offset << ", " << offset + count << ") axis=" << axis << std::endl;
                } else {
                    // Graceful degradation instead of a hard failure: keep the node replicated
                    // so numerics remain correct (documented in STATE.md, S07).
                    seam.kind = TPSeamKind::REPLICA;
                    seam.collective = TPCollectiveKind::NONE;
                    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform keeps N_SPLIT node replicated (rank " << m_rank
                                   << "): " << node->get_friendly_name() << " N=" << n_out
                                   << " axis=" << axis << " divisible=" << divisible
                                   << " safe_consumers=" << safe_consumers << std::endl;
                }
            } else {
                seam.kind = TPSeamKind::REPLICA;
                seam.collective = TPCollectiveKind::NONE;
            }
        }

        GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform seam: " << tp_seam_to_string(seam) << std::endl;
        seams.push_back(std::move(seam));
    }

    // Model outputs that are produced by a sliced N-split node (possibly through purely
    // element-wise ops) require an all-gather equivalent on rank 0 (S07).
    // ov::Model::outputs() exposes the Results in get_results() order, so the result index
    // is the output index.
    if (graph_modified) {
        const auto results = model->get_results();
        for (size_t idx = 0; idx < results.size(); ++idx) {
            std::vector<std::shared_ptr<ov::Node>> stack;
            if (results[idx]->get_input_size() > 0) {
                stack.push_back(results[idx]->input_value(0).get_node_shared_ptr());
            }
            std::unordered_set<const ov::Node*> visited;
            std::shared_ptr<ov::Node> hit;
            while (!stack.empty() && !hit) {
                auto cur = stack.back();
                stack.pop_back();
                if (!visited.insert(cur.get()).second) {
                    continue;
                }
                if (sliced_nodes.count(cur.get()) != 0) {
                    hit = cur;
                    break;
                }
                if (is_elementwise(cur)) {
                    for (const auto& in : cur->input_values()) {
                        stack.push_back(in.get_node_shared_ptr());
                    }
                }
            }
            if (hit) {
                int64_t axis = -1;
                int64_t full = -1;
                for (const auto& entry : sliced_info) {
                    if (entry.first.get() == hit.get()) {
                        axis = entry.second.first;
                        full = entry.second.second;
                        break;
                    }
                }
                TPOutputSeam os;
                os.output_index = idx;
                os.split_axis = axis;
                os.shard_dim_size = full;
                os.producer_node = hit->get_friendly_name();
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform all-gather output[" << idx << "]: "
                               << tp_output_seam_to_string(os) << std::endl;
                output_seams.push_back(std::move(os));
            }
        }
    }

    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform detected " << seams.size() << " candidate node(s), sliced "
                   << sliced_nodes.size() << ", all-gather outputs " << output_seams.size() << std::endl;

    set_tp_seams(model, std::move(seams));
    set_tp_output_seams(model, std::move(output_seams));
    return graph_modified;
}

}  // namespace ov::intel_gpu
