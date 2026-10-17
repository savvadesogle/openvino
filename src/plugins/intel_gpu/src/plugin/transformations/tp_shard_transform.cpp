// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06/S07/S08] Experimental GPU tensor-parallel support.
// TPShardTransform implementation:
//  - S06: seam detection + conservative classification (identity w.r.t. the graph);
//  - S07: real N-split - slice the constant weights (and bias) of N_SPLIT GEMMs along the
//    output-channel (N) axis for the current rank, so only this rank's slice is uploaded,
//    and record the model outputs that must be all-gathered on rank 0.
//  - S08: real K-split - slice the constant weights of K_SPLIT GEMMs along the contraction
//    (K) axis AND slice the activation operand on its last axis, so each rank computes a
//    partial GEMM sum; record K-split model outputs that must be all-reduced (summed) on
//    rank 0 with fp32 accumulation. Replicated bias/residual terms (a constant or a model
//    input folded by a residual Add) are recorded so the reducer adds them exactly ONCE.
//  - S09: int4_ASYM (group quantized). Before MarkDequantization/FC conversion the int4 weights
//    appear as a dequantization subgraph (weight u4 [.., N, G, gs] -> Convert -> Subtract(zp) ->
//    Multiply(scale) -> Reshape([.., N, K]) -> Convert). The pass recognizes this form and slices
//    weight + scale + zero_point together (and a bias Add constant for N-split) along group
//    boundaries: N-split keeps whole out-channel chunks, K-split keeps whole groups (K multiple of
//    group_size * num_ranks). Sub-byte (u4/i4) slices are byte-aligned or the node falls back to
//    REPLICA. See docs/notes/int4-asym-weights.md.
//  - S10: state (ReadValue/Assign). Variables whose state is a static, head-split tensor and whose
//    whole ReadValue -> (element-wise / broadcast) -> Assign chain is head-local are turned into
//    HEAD_SHARDED variables: each rank's variable shape is reduced to full_dim / num_ranks along the
//    head axis, and TPInferRequest concatenates the per-rank states back on query_state() (and
//    scatters on set_state()). Everything else (dynamic shapes, non-head-local chains - e.g. the
//    Qwen3.5 SDPA KV cache and the linear-attention conv/ssm states) stays REPLICATED (rank 0
//    authoritative). See docs/notes/state-inventory.md and PLAN §4.3/§5 (S10).
// See PLAN.md §2, §4.2, §4.3, §7.4.

#include "tp_shard_transform.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
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
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/variable.hpp"

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

// ---------------------------------------------------------------------------------------------
// [S09] INT4_ASYM (group-quantized) weight recognition.
//
// Before MarkDequantization / FC conversion the int4 weights of a linear projection appear as a
// dequantization subgraph (the on-disk model is plain OV IR), e.g.:
//   weight(u4 [N,G,gs]) -> Convert(f16) -> Subtract(zp[f16 N,G,1])
//                       -> Multiply(scale[f16 N,G,1]) -> Reshape([N,K]) -> Convert(f32) -> MatMul(B)
// with group_size gs=128, G=K/gs groups. The pass must recognize this form (input 1 is not a
// bare Constant) and slice weight/scale/zero_point together along group boundaries so per-rank
// slices remain a valid INT4_ASYM quantization (PLAN §2, §4.2, §4.3). See int4-asym-weights.md.
// ---------------------------------------------------------------------------------------------

//! True for the low-bit integer element types used for compressed (quantized) weights.
bool is_quantized_weight_type(const ov::element::Type& et) {
    return et.is_integral() && !et.is_dynamic() && et.bitwidth() > 0 && et.bitwidth() <= 8;
}

//! Unwraps a Constant that may be preceded by a single Convert (the pattern wraps scale/zp in a
//! Convert in some variants). Returns nullptr otherwise.
std::shared_ptr<ov::op::v0::Constant> unwrap_constant(const std::shared_ptr<ov::Node>& node) {
    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        return c;
    }
    if (ov::is_type<ov::op::v0::Convert>(node)) {
        return ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(0).get_node_shared_ptr());
    }
    return nullptr;
}

//! A recognized group-quantized (INT4_ASYM) weight subgraph feeding a GEMM's B operand.
struct QuantizedWeight {
    std::shared_ptr<ov::op::v0::Constant> weight;      //!< u4/i4/u2/... constant [.., N, G, gs]
    std::shared_ptr<ov::op::v0::Constant> scale;       //!< floating scale constant (may be null)
    std::shared_ptr<ov::op::v0::Constant> zero_point;  //!< integer zero-point constant (may be null)
    std::shared_ptr<ov::op::v1::Reshape> reshape;      //!< [.., N, G, gs] -> [.., N, K] (nullable)
    ov::Shape weight_shape;
    size_t group_axis = 0;  //!< axis of the group index G in the weight shape
    size_t group_size = 0;  //!< gs (innermost grouping dimension)
    size_t num_groups = 0;  //!< G
};

//! Matches the linear dequantization chain feeding @p gemm input 1 (see the comment above).
bool match_quantized_weight(const std::shared_ptr<ov::Node>& gemm, QuantizedWeight& qw) {
    if (gemm->get_input_size() < 2) {
        return false;
    }
    auto cur = gemm->input_value(1).get_node_shared_ptr();
    for (int guard = 0; cur && guard < 16; ++guard) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(cur)) {
            if (!is_quantized_weight_type(c->get_element_type())) {
                return false;
            }
            qw.weight = c;
            qw.weight_shape = c->get_shape();
            break;
        }
        if (ov::is_type<ov::op::v0::Convert>(cur)) {
            cur = cur->input_value(0).get_node_shared_ptr();
            continue;
        }
        if (auto r = ov::as_type_ptr<ov::op::v1::Reshape>(cur)) {
            if (!qw.reshape) {
                qw.reshape = r;
            }
            cur = cur->input_value(0).get_node_shared_ptr();
            continue;
        }
        if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(cur)) {
            auto sc = unwrap_constant(mul->input_value(1).get_node_shared_ptr());
            if (sc) {
                cur = mul->input_value(0).get_node_shared_ptr();
            } else {
                sc = unwrap_constant(mul->input_value(0).get_node_shared_ptr());
                if (!sc) {
                    return false;
                }
                cur = mul->input_value(1).get_node_shared_ptr();
            }
            if (!qw.scale) {
                qw.scale = sc;
            }
            continue;
        }
        if (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(cur)) {
            auto zc = unwrap_constant(sub->input_value(1).get_node_shared_ptr());
            if (!zc) {
                return false;
            }
            if (!qw.zero_point) {
                qw.zero_point = zc;
            }
            cur = sub->input_value(0).get_node_shared_ptr();
            continue;
        }
        return false;  // Transpose or any other op -> unsupported layout (fallback)
    }
    if (!qw.weight) {
        return false;
    }
    // Group quantization only: weight layout must be [.., N, G, gs] with G == K / gs.
    if (qw.weight_shape.size() < 3) {
        return false;
    }
    qw.group_axis = qw.weight_shape.size() - 2;
    qw.group_size = qw.weight_shape.back();
    if (qw.group_size == 0) {
        return false;
    }
    qw.num_groups = qw.weight_shape[qw.group_axis];
    return true;
}

//! The weight matrix B is operand 1 for both MatMul and FullyConnected. Accepts both a bare
//! constant and an int4_ASYM dequantization subgraph (S09).
bool has_constant_weights(const std::shared_ptr<ov::Node>& node) {
    if (node->get_input_size() < 2) {
        return false;
    }
    if (ov::is_type<ov::op::v0::Constant>(node->input_value(1).get_node())) {
        return true;
    }
    QuantizedWeight qw;
    return match_quantized_weight(node, qw);
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

//! Axis of the weights tensor that carries the contraction dimension (K). For MatMul/FC the
//! weights are [K, N] when transpose_b is false and [N, K] when it is true.
int64_t find_weight_k_axis(const std::shared_ptr<ov::Node>& node, const ov::Shape& weights_shape) {
    if (weights_shape.size() != 2) {
        return -1;
    }
    bool transpose_b = false;
    if (!get_transpose_b(node, transpose_b)) {
        return -1;
    }
    return transpose_b ? 1 : 0;
}

//! Element types that can be bit-copied while slicing. Any static width is allowed; sub-byte
//! (u4/i4/u2) slices must additionally be byte-aligned, enforced at slice time.
bool is_sliceable_type(const ov::element::Type& et) {
    return !et.is_dynamic() && et.bitwidth() > 0;
}

//! True when slicing [offset, offset + count) along @p axis of @p shape keeps whole bytes (i.e.
//! no nibble/element pair is split). Required for sub-byte (int4/u4/u2) weights/scale/zp
//! (PLAN §2). For >=8-bit element types this is always satisfied.
bool slice_is_byte_aligned(const ov::Shape& shape,
                           int64_t axis,
                           int64_t offset,
                           int64_t count,
                           int64_t bitwidth) {
    if (axis < 0 || axis >= static_cast<int64_t>(shape.size()) || bitwidth <= 0) {
        return false;
    }
    size_t inner = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }
    const size_t row_bits = shape[axis] * inner * static_cast<size_t>(bitwidth);
    const size_t off_bits = static_cast<size_t>(offset) * inner * static_cast<size_t>(bitwidth);
    const size_t slab_bits = static_cast<size_t>(count) * inner * static_cast<size_t>(bitwidth);
    // The source row stride, the slice start and the slice slab must all fall on byte boundaries so
    // no sub-byte element pair is split (PLAN §2).
    return row_bits % 8 == 0 && off_bits % 8 == 0 && slab_bits % 8 == 0;
}

//! Slices a constant along @p axis into [offset, offset + count). A bit-copy keeps the exact bit
//! pattern (including packed u4/i4 nibbles); callers must guarantee byte alignment for sub-byte
//! types via slice_is_byte_aligned().
std::shared_ptr<ov::op::v0::Constant> slice_constant(const std::shared_ptr<ov::op::v0::Constant>& src,
                                                     int64_t axis,
                                                     int64_t offset,
                                                     int64_t count) {
    const auto et = src->get_element_type();
    const auto shape = src->get_shape();
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()),
                    "[GPU][TP] slice: invalid weights axis ", axis);
    OPENVINO_ASSERT(is_sliceable_type(et), "[GPU][TP] slice: unsupported element type ", et);
    const int64_t bitwidth = et.bitwidth();
    const size_t axis_dim = shape[axis];
    OPENVINO_ASSERT(offset >= 0 && count >= 0 && static_cast<size_t>(offset + count) <= axis_dim,
                    "[GPU][TP] slice: range [", offset, ", ", offset + count, ") out of bounds for axis dim ",
                    axis_dim);

    size_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= shape[i];
    }
    size_t inner = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }

    const size_t row_bits = axis_dim * inner * static_cast<size_t>(bitwidth);
    const size_t off_bits = static_cast<size_t>(offset) * inner * static_cast<size_t>(bitwidth);
    const size_t slab_bits = static_cast<size_t>(count) * inner * static_cast<size_t>(bitwidth);
    OPENVINO_ASSERT(off_bits % 8 == 0 && slab_bits % 8 == 0 && row_bits % 8 == 0,
                    "[GPU][TP] slice: sub-byte ", et, " slice along axis ", axis, " offset ", offset,
                    " is not byte-aligned");

    ov::Shape out_shape(shape.begin(), shape.end());
    out_shape[axis] = static_cast<size_t>(count);

    const size_t row_bytes = row_bits / 8;
    const size_t off_bytes = off_bits / 8;
    const size_t slab_bytes = slab_bits / 8;
    const auto* in = static_cast<const uint8_t*>(src->get_data_ptr());
    std::vector<uint8_t> out(outer * slab_bytes);
    for (size_t o = 0; o < outer; ++o) {
        const uint8_t* src_base = in + o * row_bytes + off_bytes;
        std::memcpy(out.data() + o * slab_bytes, src_base, slab_bytes);
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

//! Slices the last axis of a (non-constant) tensor into [start, stop) using ov::op::v8::Slice.
//! Used by K-split to cut the activation operand on the contraction dimension.
std::shared_ptr<ov::op::v8::Slice> slice_last_axis(const ov::Output<ov::Node>& data, int64_t start, int64_t stop) {
    const auto i64 = ov::element::i64;
    auto axis = ov::op::v0::Constant::create(i64, ov::Shape{1}, {-1});
    auto start_c = ov::op::v0::Constant::create(i64, ov::Shape{1}, {start});
    auto stop_c = ov::op::v0::Constant::create(i64, ov::Shape{1}, {stop});
    auto step_c = ov::op::v0::Constant::create(i64, ov::Shape{1}, {1});
    return std::make_shared<ov::op::v8::Slice>(data, start_c, stop_c, step_c, axis);
}

//! Reads a constant into an fp32 vector (row-major). Supports the numeric element types the
//! reducer can ingest; unsupported types yield an empty vector.
std::vector<float> constant_to_floats(const std::shared_ptr<ov::op::v0::Constant>& c) {
    const auto et = c->get_element_type();
    const auto shape = c->get_shape();
    const size_t n = shape.empty() ? 1 : ov::shape_size(shape);
    std::vector<float> out(n);
    if (et == ov::element::f32) {
        const auto* p = c->get_data_ptr<float>();
        std::copy(p, p + n, out.begin());
    } else if (et == ov::element::f16) {
        const auto* p = c->get_data_ptr<ov::float16>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::bf16) {
        const auto* p = c->get_data_ptr<ov::bfloat16>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::f64) {
        const auto* p = c->get_data_ptr<double>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::i64) {
        const auto* p = c->get_data_ptr<int64_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::i32) {
        const auto* p = c->get_data_ptr<int32_t>();
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
    } else {
        return {};
    }
    return out;
}

//! Row-major strides for a shape.
std::vector<size_t> row_major_strides(const ov::Shape& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

//! Broadcasts @p src (shaped @p src_shape) to @p out_shape (trailing-dim alignment, numpy
//! rules). Returns false when the shapes are not broadcast-compatible.
bool broadcast_expand(const std::vector<float>& src,
                      const ov::Shape& src_shape,
                      const ov::Shape& out_shape,
                      std::vector<float>& out) {
    const size_t out_rank = out_shape.size();
    if (src_shape.size() > out_rank) {
        return false;
    }
    ov::Shape aligned(out_rank, 1);
    for (size_t i = 0; i < src_shape.size(); ++i) {
        aligned[out_rank - src_shape.size() + i] = src_shape[i];
    }
    for (size_t i = 0; i < out_rank; ++i) {
        if (aligned[i] != out_shape[i] && aligned[i] != 1) {
            return false;
        }
    }
    const auto src_strides = row_major_strides(aligned);
    const auto out_strides = row_major_strides(out_shape);
    const size_t total = ov::shape_size(out_shape);
    out.assign(total, 0.0f);
    for (size_t idx = 0; idx < total; ++idx) {
        size_t rem = idx;
        size_t sidx = 0;
        for (size_t d = 0; d < out_rank; ++d) {
            const size_t coord = out_strides[d] == 0 ? 0 : rem / out_strides[d];
            rem = out_strides[d] == 0 ? 0 : rem % out_strides[d];
            if (aligned[d] != 1) {
                sidx += coord * src_strides[d];
            }
        }
        out[idx] = src[sidx];
    }
    return true;
}

//! Result of analyzing the reduction path of a K-split node.
struct KReduceScan {
    bool safe = true;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;
    std::vector<size_t> input_indices;
};

//! Walks the consumers of a K-split node (through element-wise ops only) and collects the
//! replicated additive terms fold into the output via Add: constants (e.g. a bias) and model
//! inputs (e.g. a residual stream). Any other replicated operand (a genuine residual tensor
//! produced inside the graph) or a non-element-wise consumer makes the node unsafe to K-split
//! in S08 (correct handling needs a device-side all-reduce placed before the Add, S09+).
void scan_k_reduction(const std::shared_ptr<ov::Node>& start,
                      const std::shared_ptr<const ov::Model>& model,
                      KReduceScan& scan) {
    std::unordered_set<const ov::Node*> visited{start.get()};
    std::vector<std::shared_ptr<ov::Node>> work{start};
    while (!work.empty()) {
        auto cur = work.back();
        work.pop_back();
        for (const auto& target : cur->output(0).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::Result>(consumer)) {
                continue;
            }
            if (ov::is_type<ov::op::v1::Add>(consumer)) {
                auto in0 = consumer->input_value(0).get_node_shared_ptr();
                auto in1 = consumer->input_value(1).get_node_shared_ptr();
                std::shared_ptr<ov::Node> other;
                if (in0.get() == cur.get()) {
                    other = in1;
                } else if (in1.get() == cur.get()) {
                    other = in0;
                }
                if (!other) {
                    scan.safe = false;
                    continue;
                }
                if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(other)) {
                    scan.constants.push_back(c);
                } else if (auto p = ov::as_type_ptr<ov::op::v0::Parameter>(other)) {
                    size_t idx = 0;
                    bool found = false;
                    for (const auto& param : model->get_parameters()) {
                        if (param.get() == p.get()) {
                            found = true;
                            break;
                        }
                        ++idx;
                    }
                    if (found) {
                        scan.input_indices.push_back(idx);
                    } else {
                        scan.safe = false;
                    }
                } else {
                    scan.safe = false;
                }
                if (visited.insert(consumer.get()).second) {
                    work.push_back(consumer);
                }
            } else if (is_elementwise(consumer)) {
                if (visited.insert(consumer.get()).second) {
                    work.push_back(consumer);
                }
            } else {
                scan.safe = false;
            }
        }
    }
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

//! Result of analyzing the downstream (element-wise) consumers of an N-split node.
struct NSplitScan {
    bool safe = true;
    int64_t n_out = -1;
    // Add-constants (bias) reachable through element-wise ops that must be sliced along the last
    // (out-channel) axis together with the weights.
    std::vector<std::shared_ptr<ov::op::v0::Constant>> side_constants;
};

//! Walks the downstream element-wise consumers of an N-split node. The node may only be N-split if
//! the assembled slice never feeds a further GEMM (which would need the full N: requires a real
//! inter-node all-gather, out of scope) and any residual Add operand is either a constant bias
//! (sliced) or broadcast-compatible. Any non-element-wise node (a GEMM) or a full-width
//! non-constant Add operand makes the node unsafe (falls back to REPLICA).
void scan_nsplit_consumers(const std::shared_ptr<ov::Node>& start, int64_t n_out, NSplitScan& scan) {
    scan.n_out = n_out;
    std::unordered_set<const ov::Node*> visited{start.get()};
    std::vector<std::shared_ptr<ov::Node>> work{start};
    while (!work.empty()) {
        auto cur = work.back();
        work.pop_back();
        for (const auto& target : cur->output(0).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::Result>(consumer)) {
                continue;
            }
            if (!is_elementwise(consumer)) {
                scan.safe = false;  // a GEMM (or other non-element-wise op) needs the full N
                continue;
            }
            if (ov::is_type<ov::op::v1::Add>(consumer)) {
                auto in0 = consumer->input_value(0).get_node_shared_ptr();
                auto in1 = consumer->input_value(1).get_node_shared_ptr();
                std::shared_ptr<ov::Node> other;
                if (in0.get() == cur.get()) {
                    other = in1;
                } else if (in1.get() == cur.get()) {
                    other = in0;
                }
                if (other) {
                    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(other)) {
                        const auto shape = c->get_shape();
                        if (!shape.empty() && shape.back() == static_cast<size_t>(n_out)) {
                            scan.side_constants.push_back(c);
                        }
                    } else {
                        const auto other_pshape = other->get_output_partial_shape(0);
                        if (other_pshape.rank().is_static() && other_pshape.rank().get_length() >= 1) {
                            const auto& last = other_pshape[other_pshape.rank().get_length() - 1];
                            if (last.is_static() && last.get_length() == n_out) {
                                scan.safe = false;  // full-width residual: needs device-side handling
                            }
                        }
                    }
                }
            }
            if (visited.insert(consumer.get()).second) {
                work.push_back(consumer);
            }
        }
    }
}

//! Replaces every consumer of a constant with @p new_c and transfers naming/rt_info.
void replace_constant(const std::shared_ptr<ov::op::v0::Constant>& old_c,
                      const std::shared_ptr<ov::op::v0::Constant>& new_c,
                      const std::string& suffix) {
    new_c->set_friendly_name(old_c->get_friendly_name() + suffix);
    ov::copy_runtime_info(old_c, new_c);
    ov::copy_weightless_cache_attr(old_c, new_c);
    for (auto& target : old_c->output(0).get_target_inputs()) {
        target.replace_source_output(new_c->output(0));
    }
}

//! Index of the first axis of @p shape equal to @p dim (preferring @p preferred); -1 if none.
int64_t find_axis_of(const ov::Shape& shape, size_t dim, size_t preferred) {
    if (preferred < shape.size() && shape[preferred] == dim) {
        return static_cast<int64_t>(preferred);
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == dim) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

//! Prepares an updated Reshape shape constant replacing @p old_dim with @p new_dim (no mutation).
std::shared_ptr<ov::op::v0::Constant> prepare_shape_update(const std::shared_ptr<ov::op::v1::Reshape>& reshape,
                                                           int64_t old_dim,
                                                           int64_t new_dim) {
    if (!reshape) {
        return nullptr;
    }
    auto sc = ov::as_type_ptr<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
    if (!sc || sc->get_element_type() != ov::element::i64) {
        return nullptr;
    }
    auto vals = sc->cast_vector<int64_t>();
    bool found = false;
    for (auto& v : vals) {
        if (v == old_dim) {
            v = new_dim;
            found = true;
            break;
        }
    }
    if (!found) {
        return nullptr;
    }
    auto nc = ov::op::v0::Constant::create(ov::element::i64, sc->get_shape(), vals);
    ov::copy_runtime_info(sc, nc);
    return nc;
}

//! Prepares the byte-aligned slice of an optional scale/zp constant along @p axis (skips per-tensor
//! constants with a single element, which must not be sliced). Returns false if unsliceable.
bool prepare_side_slice(const std::shared_ptr<ov::op::v0::Constant>& c,
                        int64_t axis,
                        int64_t offset,
                        int64_t count,
                        std::shared_ptr<ov::op::v0::Constant>& out) {
    if (!c) {
        out = nullptr;
        return true;
    }
    if (ov::shape_size(c->get_shape()) == 1) {
        out = nullptr;  // per-tensor: keep as-is (never sliced, PLAN §2 / cpu-tp-pattern §5)
        return true;
    }
    if (axis < 0 || !is_sliceable_type(c->get_element_type()) ||
        !slice_is_byte_aligned(c->get_shape(), axis, offset, count, c->get_element_type().bitwidth())) {
        return false;
    }
    out = slice_constant(c, axis, offset, count);
    return true;
}

//! N-split of a bare-constant weight: slice the N axis + matching bias Add/FC-bias constants.
bool apply_nsplit_bare(const std::shared_ptr<ov::Node>& node,
                       const std::shared_ptr<ov::op::v0::Constant>& weights,
                       int64_t n_out,
                       int64_t offset,
                       int64_t count,
                       const NSplitScan& scan) {
    const int64_t axis = find_weight_split_axis(node, weights->get_shape(), n_out);
    if (axis < 0 || weights->get_shape()[axis] != static_cast<size_t>(n_out) ||
        !is_sliceable_type(weights->get_element_type()) ||
        !slice_is_byte_aligned(weights->get_shape(), axis, offset, count, weights->get_element_type().bitwidth())) {
        return false;
    }
    std::vector<std::pair<std::shared_ptr<ov::op::v0::Constant>, std::shared_ptr<ov::op::v0::Constant>>> sides;
    for (const auto& c : scan.side_constants) {
        const auto shape = c->get_shape();
        if (shape.empty() || shape.back() != static_cast<size_t>(n_out)) {
            return false;
        }
        const int64_t a = static_cast<int64_t>(shape.size()) - 1;
        if (!slice_is_byte_aligned(shape, a, offset, count, c->get_element_type().bitwidth())) {
            return false;
        }
        sides.emplace_back(c, slice_constant(c, a, offset, count));
    }
    auto sliced_weights = slice_constant(weights, axis, offset, count);
    replace_constant(weights, sliced_weights, "_tp_rank");
    slice_bias_if_present(node, n_out, offset, count);
    for (auto& s : sides) {
        replace_constant(s.first, s.second, "_tp_nrank");
    }
    return true;
}

//! N-split of an int4_ASYM dequant subgraph: slice weight + scale + zp along the N axis (whole
//! out-channel chunks), update the Reshape target and bias Add constants.
bool apply_nsplit_quantized(const std::shared_ptr<ov::Node>& node,
                            const QuantizedWeight& qw,
                            int64_t n_out,
                            int64_t offset,
                            int64_t count,
                            const NSplitScan& scan) {
    const auto& wshape = qw.weight_shape;
    const int64_t n_axis = find_axis_of(wshape, static_cast<size_t>(n_out), 0);
    if (n_axis < 0 || n_axis == static_cast<int64_t>(qw.group_axis) ||
        n_axis == static_cast<int64_t>(wshape.size()) - 1 ||
        !is_sliceable_type(qw.weight->get_element_type()) ||
        !slice_is_byte_aligned(wshape, n_axis, offset, count, qw.weight->get_element_type().bitwidth())) {
        return false;
    }
    std::shared_ptr<ov::op::v0::Constant> sliced_scale, sliced_zp;
    const int64_t scale_axis = qw.scale ? find_axis_of(qw.scale->get_shape(), static_cast<size_t>(n_out), 0) : -1;
    const int64_t zp_axis = qw.zero_point ? find_axis_of(qw.zero_point->get_shape(), static_cast<size_t>(n_out), 0) : -1;
    if (!prepare_side_slice(qw.scale, scale_axis, offset, count, sliced_scale) ||
        !prepare_side_slice(qw.zero_point, zp_axis, offset, count, sliced_zp)) {
        return false;
    }
    std::shared_ptr<ov::op::v0::Constant> new_shape_const;
    if (qw.reshape) {
        new_shape_const = prepare_shape_update(qw.reshape, n_out, count);
        if (!new_shape_const) {
            return false;
        }
    }
    std::vector<std::pair<std::shared_ptr<ov::op::v0::Constant>, std::shared_ptr<ov::op::v0::Constant>>> sides;
    for (const auto& c : scan.side_constants) {
        const auto shape = c->get_shape();
        if (shape.empty() || shape.back() != static_cast<size_t>(n_out)) {
            return false;
        }
        const int64_t a = static_cast<int64_t>(shape.size()) - 1;
        if (!slice_is_byte_aligned(shape, a, offset, count, c->get_element_type().bitwidth())) {
            return false;
        }
        sides.emplace_back(c, slice_constant(c, a, offset, count));
    }
    auto sliced_weights = slice_constant(qw.weight, n_axis, offset, count);
    replace_constant(qw.weight, sliced_weights, "_tp_rank");
    if (sliced_scale) {
        replace_constant(qw.scale, sliced_scale, "_tp_nrank");
    }
    if (sliced_zp) {
        replace_constant(qw.zero_point, sliced_zp, "_tp_nrank");
    }
    if (qw.reshape && new_shape_const) {
        qw.reshape->input(1).replace_source_output(new_shape_const);
    }
    for (auto& s : sides) {
        replace_constant(s.first, s.second, "_tp_nrank");
    }
    return true;
}

//! K-split of a bare-constant 2D weight: slice the contraction axis.
bool apply_ksplit_bare(const std::shared_ptr<ov::op::v0::Constant>& weights,
                       int64_t k_axis,
                       int64_t offset,
                       int64_t count) {
    if (k_axis < 0 || !is_sliceable_type(weights->get_element_type()) ||
        !slice_is_byte_aligned(weights->get_shape(), k_axis, offset, count, weights->get_element_type().bitwidth())) {
        return false;
    }
    auto sliced = slice_constant(weights, k_axis, offset, count);
    replace_constant(weights, sliced, "_tp_krank");
    return true;
}

//! K-split of an int4_ASYM dequant subgraph: slice weight + scale + zp along the group axis (whole
//! groups) and update the Reshape target. Each rank keeps G/num_ranks groups (K multiple of
//! group_size * num_ranks).
bool apply_ksplit_quantized(const QuantizedWeight& qw, int64_t offset_g, int64_t count_g) {
    const auto& wshape = qw.weight_shape;
    const int64_t ga = static_cast<int64_t>(qw.group_axis);
    if (!is_sliceable_type(qw.weight->get_element_type()) ||
        !slice_is_byte_aligned(wshape, ga, offset_g, count_g, qw.weight->get_element_type().bitwidth())) {
        return false;
    }
    std::shared_ptr<ov::op::v0::Constant> sliced_scale, sliced_zp;
    const int64_t scale_axis = qw.scale ? find_axis_of(qw.scale->get_shape(), qw.num_groups, 0) : -1;
    const int64_t zp_axis = qw.zero_point ? find_axis_of(qw.zero_point->get_shape(), qw.num_groups, 0) : -1;
    if (!prepare_side_slice(qw.scale, scale_axis, offset_g, count_g, sliced_scale) ||
        !prepare_side_slice(qw.zero_point, zp_axis, offset_g, count_g, sliced_zp)) {
        return false;
    }
    const int64_t k_dim = static_cast<int64_t>(qw.num_groups * qw.group_size);
    const int64_t new_k = static_cast<int64_t>(count_g * qw.group_size);
    std::shared_ptr<ov::op::v0::Constant> new_shape_const;
    if (qw.reshape) {
        new_shape_const = prepare_shape_update(qw.reshape, k_dim, new_k);
        if (!new_shape_const) {
            return false;
        }
    }
    auto sliced_weights = slice_constant(qw.weight, ga, offset_g, count_g);
    replace_constant(qw.weight, sliced_weights, "_tp_krank");
    if (sliced_scale) {
        replace_constant(qw.scale, sliced_scale, "_tp_krank");
    }
    if (sliced_zp) {
        replace_constant(qw.zero_point, sliced_zp, "_tp_krank");
    }
    if (qw.reshape && new_shape_const) {
        qw.reshape->input(1).replace_source_output(new_shape_const);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// [S10] State (ReadValue/Assign) head-sharding analysis. See docs/notes/state-inventory.md.
// ---------------------------------------------------------------------------------------------

//! Preferred head axis of a state tensor: the axis adjacent to the innermost one (per-head layout
//! [.., H, hd], e.g. the KV cache [batch, kv_heads, seq, head_dim]); falls back to the first
//! divisible interior axis. Returns -1 when no divisible interior axis exists.
int64_t find_state_head_axis(const ov::Shape& shape, int32_t num_ranks) {
    const int64_t rank = static_cast<int64_t>(shape.size());
    if (rank < 2 || num_ranks <= 1) {
        return -1;
    }
    auto divisible = [&](int64_t a) {
        return a >= 1 && a <= rank - 2 && shape[a] > 1 && (shape[a] % static_cast<size_t>(num_ranks) == 0);
    };
    const int64_t preferred = rank - 2;
    if (divisible(preferred)) {
        return preferred;
    }
    for (int64_t a = 1; a <= rank - 2; ++a) {
        if (divisible(a)) {
            return a;
        }
    }
    return -1;
}

//! True when an operand of an element-wise node may keep its shape when the state head axis is
//! reduced to full_dim / num_ranks, i.e. it broadcasts along that axis.
bool operand_broadcasts_at_axis(const ov::PartialShape& pshape, int64_t axis) {
    if (pshape.rank().is_dynamic()) {
        return false;
    }
    const int64_t r = pshape.rank().get_length();
    if (r <= axis) {
        return true;  // right-aligned: no dimension on the head axis -> broadcast
    }
    const auto& dim = pshape[axis];
    return dim.is_static() && dim.get_length() == 1;
}

//! Result of analyzing one state variable for head-sharding.
struct StateShardAnalysis {
    bool safe = false;
    int64_t axis = -1;
    int64_t full_dim = -1;
    ov::Shape full_shape;
    std::vector<size_t> derived_results;  // model output indices reachable via element-wise ops
    std::string reason;
};

//! State variable id of a ReadValue/Assign node ("" for other nodes).
std::string state_variable_id(const std::shared_ptr<ov::Node>& node) {
    if (auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
        return rv->get_variable_id();
    }
    if (auto as = ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
        return as->get_variable_id();
    }
    return {};
}

//! Analyzes whether @p var can be head-sharded: static, numeric, divisible head axis, and a chain
//! ReadValue -> (element-wise/broadcast) -> Assign that is entirely head-local.
StateShardAnalysis analyze_state_variable(const std::shared_ptr<ov::op::util::Variable>& var,
                                          const std::vector<std::shared_ptr<ov::Node>>& read_values,
                                          const std::vector<std::shared_ptr<ov::Node>>& assigns,
                                          const std::unordered_map<const ov::Node*, size_t>& result_index,
                                          int32_t num_ranks) {
    StateShardAnalysis a;
    const auto info = var->get_info();
    if (!info.data_shape.is_static()) {
        a.reason = "dynamic state shape";
        return a;
    }
    if (info.data_type.is_dynamic() || info.data_type == ov::element::string ||
        info.data_type == ov::element::boolean) {
        a.reason = "non-numeric state type";
        return a;
    }
    a.full_shape = info.data_shape.to_shape();
    a.axis = find_state_head_axis(a.full_shape, num_ranks);
    if (a.axis < 0) {
        a.reason = "head axis not divisible by num_ranks";
        return a;
    }
    a.full_dim = static_cast<int64_t>(a.full_shape[a.axis]);

    // A ReadValue may carry an initialization subgraph; its value must broadcast along the head
    // axis so the (sharded) variable shape still relaxes it. Otherwise the state stays replicated.
    for (const auto& rv : read_values) {
        if (rv->get_input_size() > 0 && !operand_broadcasts_at_axis(rv->get_input_partial_shape(0), a.axis)) {
            a.reason = "non-broadcast ReadValue initializer on head axis";
            return a;
        }
    }

    const std::string target_id = info.variable_id;
    std::unordered_set<const ov::Node*> reachable;
    std::vector<std::shared_ptr<ov::Node>> work;
    std::vector<std::shared_ptr<ov::Node>> elementwise_nodes;
    for (const auto& rv : read_values) {
        reachable.insert(rv.get());
        work.push_back(rv);
    }
    std::unordered_set<const ov::Node*> covered_assigns;
    while (!work.empty()) {
        auto cur = work.back();
        work.pop_back();
        for (size_t oi = 0; oi < cur->get_output_size(); ++oi) {
            for (const auto& target : cur->output(oi).get_target_inputs()) {
                auto consumer = target.get_node()->shared_from_this();
                if (auto res = ov::as_type_ptr<ov::op::v0::Result>(consumer)) {
                    auto it = result_index.find(res.get());
                    if (it != result_index.end()) {
                        a.derived_results.push_back(it->second);
                    }
                    continue;
                }
                if (ov::is_type<ov::op::util::AssignBase>(consumer)) {
                    if (state_variable_id(consumer) != target_id) {
                        a.reason = "Assign of a different variable in chain";
                        return a;
                    }
                    covered_assigns.insert(consumer.get());
                    continue;
                }
                if (!is_elementwise(consumer)) {
                    a.reason = "non-element-wise state consumer";
                    return a;
                }
                if (reachable.insert(consumer.get()).second) {
                    work.push_back(consumer);
                    elementwise_nodes.push_back(consumer);
                }
            }
        }
    }
    // Second pass: every operand of every reached element-wise node must be a matching ReadValue,
    // another reached node, or broadcast along the head axis.
    for (const auto& n : elementwise_nodes) {
        for (const auto& iv : n->input_values()) {
            auto producer = iv.get_node_shared_ptr();
            if (ov::is_type<ov::op::util::ReadValueBase>(producer)) {
                if (state_variable_id(producer) != target_id) {
                    a.reason = "state chain mixes variables";
                    return a;
                }
                continue;
            }
            if (reachable.count(producer.get())) {
                continue;
            }
            if (!operand_broadcasts_at_axis(producer->get_output_partial_shape(iv.get_index()), a.axis)) {
                a.reason = "non-broadcast operand on head axis";
                return a;
            }
        }
    }
    for (const auto& as : assigns) {
        if (!covered_assigns.count(as.get())) {
            a.reason = "Assign not fed by matching ReadValue";
            return a;
        }
    }
    std::sort(a.derived_results.begin(), a.derived_results.end());
    a.derived_results.erase(std::unique(a.derived_results.begin(), a.derived_results.end()),
                            a.derived_results.end());
    a.safe = true;
    return a;
}

//! A model output that is produced by a HEAD_SHARDED state chain (all-gather along @p axis).
struct StateOutputSeam {
    size_t output_index = 0;
    int64_t axis = -1;
    int64_t full_dim = -1;
    int64_t full_rank = 0;
    std::string variable_id;
};

}  // namespace

bool TPShardTransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const bool sharding_active = m_num_ranks > 1 && m_rank >= 0 && m_rank < m_num_ranks;

    // Geometry of a node that was actually sliced for this rank.
    struct SlicedNodeInfo {
        bool is_k = false;
        int64_t split_axis = -1;        // N axis (N-split) or K axis (K-split); informational
        int64_t shard_dim_size = -1;    // full size of the sharded dim (N for N-split, K for K-split)
        std::vector<TPReplicatedAdd> replicated_adds;  // K-split only
    };

    std::vector<TPSeam> seams;
    std::vector<TPOutputSeam> output_seams;
    // Nodes whose output was actually sliced for this rank.
    std::unordered_map<const ov::Node*, SlicedNodeInfo> sliced_nodes;

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
            const auto bare_weights = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
            QuantizedWeight qw;
            const bool quantized = !bare_weights && match_quantized_weight(node, qw);
            const auto out_pshape = node->get_output_partial_shape(0);
            bool sliced = false;
            int64_t n_out = -1;
            bool divisible = false;
            if ((bare_weights || quantized) && out_pshape.rank().is_static() &&
                out_pshape.rank().get_length() >= 1 &&
                out_pshape[out_pshape.rank().get_length() - 1].is_static()) {
                n_out = out_pshape[out_pshape.rank().get_length() - 1].get_length();
                divisible = (n_out > 0) && (n_out % m_num_ranks == 0);
                if (divisible) {
                    const int64_t count = n_out / m_num_ranks;
                    const int64_t offset = static_cast<int64_t>(m_rank) * count;
                    NSplitScan scan;
                    scan_nsplit_consumers(node, n_out, scan);
                    if (scan.safe) {
                        if (bare_weights) {
                            sliced = apply_nsplit_bare(node, bare_weights, n_out, offset, count, scan);
                        } else {
                            sliced = apply_nsplit_quantized(node, qw, n_out, offset, count, scan);
                        }
                    }
                }
            }
            if (sliced) {
                seam.split_axis = -1;  // N is the last dimension of the GEMM output
                seam.shard_dim_size = n_out;

                SlicedNodeInfo info;
                info.is_k = false;
                info.split_axis = seam.split_axis;
                info.shard_dim_size = seam.shard_dim_size;
                sliced_nodes[node.get()] = std::move(info);
                graph_modified = true;
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform N-split rank " << m_rank << "/" << m_num_ranks
                               << " " << node->get_friendly_name() << ": N=" << n_out
                               << " quantized=" << quantized << " slice=[" << m_rank * (n_out / m_num_ranks)
                               << ", " << (m_rank + 1) * (n_out / m_num_ranks) << ")" << std::endl;
            } else {
                // Graceful degradation instead of a hard failure: keep the node replicated
                // so numerics remain correct (documented in STATE.md, S07/S09).
                seam.kind = TPSeamKind::REPLICA;
                seam.collective = TPCollectiveKind::NONE;
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform keeps N_SPLIT node replicated (rank " << m_rank
                               << "): " << node->get_friendly_name() << " N=" << n_out
                               << " quantized=" << quantized << " divisible=" << divisible << std::endl;
            }
        } else if (sharding_active && seam.kind == TPSeamKind::K_SPLIT) {
            // [S08] K-split: each rank computes a partial GEMM sum over its slice of the
            // contraction dimension; the reducer sums the partials (all-reduce sum, fp32).
            const auto bare_weights = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
            QuantizedWeight qw;
            const bool quantized = !bare_weights && match_quantized_weight(node, qw);
            const auto in0_pshape = node->get_input_partial_shape(0);
            const auto out_pshape = node->get_output_partial_shape(0);

            int64_t k_axis = -1;
            int64_t k_dim = -1;
            int64_t group_size = 1;
            if (bare_weights) {
                k_axis = find_weight_k_axis(node, bare_weights->get_shape());
                if (k_axis >= 0) {
                    k_dim = static_cast<int64_t>(bare_weights->get_shape()[k_axis]);
                }
            } else if (quantized) {
                k_axis = static_cast<int64_t>(qw.group_axis);
                group_size = static_cast<int64_t>(qw.group_size);
                k_dim = static_cast<int64_t>(qw.num_groups * qw.group_size);
            }
            int64_t act_k = -1;
            if (in0_pshape.rank().is_static() && in0_pshape.rank().get_length() >= 1) {
                const auto& last = in0_pshape[in0_pshape.rank().get_length() - 1];
                if (last.is_static()) {
                    act_k = last.get_length();
                }
            }
            // The contraction dimension is the last axis of A only when A is not transposed.
            bool transpose_a = false;
            if (auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
                transpose_a = mm->get_transpose_a();
            }
            const bool out_static = out_pshape.is_static();
            // int4_ASYM: the contraction axis must be sliced whole groups (K multiple of
            // group_size * num_ranks), i.e. num_groups divisible by num_ranks (PLAN §2).
            const bool divisible = quantized ? (qw.num_groups > 0 && (qw.num_groups % m_num_ranks == 0))
                                             : ((k_dim > 0) && (k_dim % m_num_ranks == 0));
            const bool shape_ok = !transpose_a && act_k == k_dim && out_static &&
                                  ((bare_weights && k_axis >= 0 && is_sliceable_type(bare_weights->get_element_type())) ||
                                   (quantized && is_sliceable_type(qw.weight->get_element_type())));

            KReduceScan scan;
            if (shape_ok && divisible) {
                scan_k_reduction(node, model, scan);
            } else {
                scan.safe = false;
            }

            // Validate that every replicated constant can be broadcast to the output shape
            // before touching the graph.
            std::vector<TPReplicatedAdd> replicated_adds;
            bool adds_ok = true;
            if (scan.safe && out_static) {
                const auto out_shape = node->get_output_shape(0);
                auto push_constant_add = [&](const std::shared_ptr<ov::op::v0::Constant>& c) {
                    auto vals = constant_to_floats(c);
                    std::vector<float> expanded;
                    if (vals.empty() || !broadcast_expand(vals, c->get_shape(), out_shape, expanded)) {
                        return false;
                    }
                    TPReplicatedAdd add;
                    add.kind = TPReplicatedAddKind::CONSTANT;
                    add.values = std::move(expanded);
                    replicated_adds.push_back(std::move(add));
                    return true;
                };
                for (const auto& c : scan.constants) {
                    if (!push_constant_add(c)) {
                        adds_ok = false;
                        break;
                    }
                }
                // FullyConnected carries its bias as operand 2 (not a separate Add): it is
                // replicated on every rank and must be counted once by the reducer (never sliced).
                if (adds_ok && node->get_input_size() >= 3) {
                    if (auto bias = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(2).get_node_shared_ptr())) {
                        if (!push_constant_add(bias)) {
                            adds_ok = false;
                        }
                    }
                }
                if (adds_ok) {
                    for (size_t idx : scan.input_indices) {
                        TPReplicatedAdd add;
                        add.kind = TPReplicatedAddKind::INPUT;
                        add.input_index = idx;
                        replicated_adds.push_back(std::move(add));
                    }
                }
            }

            if (shape_ok && divisible && scan.safe && adds_ok) {
                int64_t offset = 0;
                int64_t count = 0;
                bool applied = false;
                if (bare_weights) {
                    // 1a) slice the constant weights along the contraction axis
                    count = k_dim / m_num_ranks;
                    offset = static_cast<int64_t>(m_rank) * count;
                    applied = apply_ksplit_bare(bare_weights, k_axis, offset, count);
                } else {
                    // 1b) slice the dequant subgraph along the group axis (whole groups)
                    const int64_t count_g = static_cast<int64_t>(qw.num_groups / m_num_ranks);
                    const int64_t offset_g = static_cast<int64_t>(m_rank) * count_g;
                    offset = offset_g * group_size;
                    count = count_g * group_size;
                    applied = apply_ksplit_quantized(qw, offset_g, count_g);
                }

                if (applied) {
                    // 2) slice the activation operand on its last (contraction) axis so each rank
                    //    holds the matching input slice (only its K range is uploaded/computed).
                    auto act_slice = slice_last_axis(node->input_value(0), offset, offset + count);
                    act_slice->set_friendly_name(node->get_friendly_name() + "_tp_act");
                    node->input(0).replace_source_output(act_slice->output(0));

                    seam.split_axis = k_axis;
                    seam.shard_dim_size = k_dim;

                    SlicedNodeInfo info;
                    info.is_k = true;
                    info.split_axis = k_axis;
                    info.shard_dim_size = k_dim;
                    info.replicated_adds = std::move(replicated_adds);
                    sliced_nodes[node.get()] = std::move(info);
                    graph_modified = true;
                    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform K-split rank " << m_rank << "/" << m_num_ranks
                                   << " " << node->get_friendly_name() << ": K=" << k_dim
                                   << " quantized=" << quantized << " slice=[" << offset << ", " << offset + count
                                   << ") k_axis=" << k_axis
                                   << " replicated_adds=" << scan.constants.size() << "+" << scan.input_indices.size()
                                   << std::endl;
                } else {
                    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform K-split not applied (rank " << m_rank
                                   << "): " << node->get_friendly_name()
                                   << " (unsafe sub-byte/byte alignment) -> replica" << std::endl;
                }
            } else {
                // Unsliced K-split stays a replication (each rank computes the full GEMM);
                // classification (seam.kind) is left unchanged so tp_seams stays stable.
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform keeps K_SPLIT node replicated (rank " << m_rank
                               << "): " << node->get_friendly_name() << " K=" << k_dim
                               << " divisible=" << divisible << " shape_ok=" << shape_ok
                               << " safe_reduction=" << scan.safe << " adds_ok=" << adds_ok << std::endl;
            }
        }

        GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform seam: " << tp_seam_to_string(seam) << std::endl;
        seams.push_back(std::move(seam));
    }

    // [S10] State (ReadValue/Assign) head-sharding. Must run after the GEMM pass so that the
    // final graph (weights sliced, activations sliced) is what gets validated. A state variable is
    // head-sharded only when it is static, numeric, its head axis is divisible and the whole
    // ReadValue -> (element-wise/broadcast) -> Assign chain stays head-local; otherwise it is
    // reported REPLICATED (rank 0 authoritative) with a diagnostic reason.
    std::vector<TPStateSeam> state_seams;
    std::vector<StateOutputSeam> state_output_seams;
    if (sharding_active) {
        struct VarGroup {
            std::shared_ptr<ov::op::util::Variable> var;
            std::vector<std::shared_ptr<ov::Node>> read_values;
            std::vector<std::shared_ptr<ov::Node>> assigns;
        };
        std::unordered_map<std::string, VarGroup> var_groups;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
                auto var = rv->get_variable();
                if (!var) {
                    continue;
                }
                auto& g = var_groups[var->get_info().variable_id];
                g.var = var;
                g.read_values.push_back(node);
            }
        }
        for (const auto& sink : model->get_sinks()) {
            if (auto as = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
                auto var = as->get_variable();
                if (!var) {
                    continue;
                }
                auto& g = var_groups[var->get_info().variable_id];
                g.var = var;
                g.assigns.push_back(as);
            }
        }

        std::unordered_map<const ov::Node*, size_t> result_index;
        {
            const auto results = model->get_results();
            for (size_t i = 0; i < results.size(); ++i) {
                result_index[results[i].get()] = i;
            }
        }

        std::vector<std::string> ids;
        ids.reserve(var_groups.size());
        for (const auto& kv : var_groups) {
            ids.push_back(kv.first);
        }
        std::sort(ids.begin(), ids.end());

        bool state_modified = false;
        for (const auto& id : ids) {
            auto& g = var_groups[id];
            auto analysis = analyze_state_variable(g.var, g.read_values, g.assigns, result_index, m_num_ranks);
            TPStateSeam ss;
            ss.variable_id = id;
            if (!analysis.safe) {
                ss.kind = TPStateShardKind::REPLICATED;
                ss.reason = analysis.reason;
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform state " << id << " -> REPLICATED ("
                               << analysis.reason << ")" << std::endl;
                state_seams.push_back(std::move(ss));
                continue;
            }
            ss.kind = TPStateShardKind::HEAD_SHARDED;
            ss.axis = analysis.axis;
            ss.full_dim = analysis.full_dim;
            ss.full_shape = analysis.full_shape;
            ss.reason = "head-local element-wise state chain";
            auto sharded = analysis.full_shape;
            sharded[analysis.axis] = sharded[analysis.axis] / static_cast<size_t>(m_num_ranks);
            g.var->update_data_shape(ov::PartialShape(sharded));
            for (size_t oi : analysis.derived_results) {
                StateOutputSeam so;
                so.output_index = oi;
                so.axis = analysis.axis;
                so.full_dim = analysis.full_dim;
                so.full_rank = static_cast<int64_t>(analysis.full_shape.size());
                so.variable_id = id;
                state_output_seams.push_back(std::move(so));
            }
            state_modified = true;
            GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform state " << id << " -> HEAD_SHARDED axis="
                           << analysis.axis << " full_dim=" << analysis.full_dim << " shard="
                           << sharded[analysis.axis] << " derived_outputs=" << analysis.derived_results.size()
                           << std::endl;
            state_seams.push_back(std::move(ss));
        }
        if (state_modified) {
            model->validate_nodes_and_infer_types();
            graph_modified = true;
        }
    }

    // Model outputs that are produced by a sliced GEMM (possibly through purely element-wise
    // ops) require assembly on rank 0: all-gather (N-split) or all-reduce(sum) (K-split).
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
                const auto& info = sliced_nodes[hit.get()];
                TPOutputSeam os;
                os.output_index = idx;
                os.collective = info.is_k ? TPCollectiveKind::ALL_REDUCE_SUM : TPCollectiveKind::ALL_GATHER;
                os.split_axis = -1;
                os.shard_dim_size = info.shard_dim_size;
                os.producer_node = hit->get_friendly_name();
                if (info.is_k) {
                    os.replicated_adds = info.replicated_adds;
                }
                GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform output[" << idx << "]: "
                               << tp_output_seam_to_string(os) << std::endl;
                output_seams.push_back(std::move(os));
            }
        }
    }

    // [S10] Outputs fed by a HEAD_SHARDED state chain also need all-gather (concatenation) along
    // the head axis (unless a sliced GEMM already claimed the same output).
    for (const auto& so : state_output_seams) {
        bool already = false;
        for (const auto& os : output_seams) {
            if (os.output_index == so.output_index) {
                already = true;
                break;
            }
        }
        if (already) {
            continue;
        }
        TPOutputSeam os;
        os.output_index = so.output_index;
        os.collective = TPCollectiveKind::ALL_GATHER;
        // Negative axis counted from the end (consistent with the N-split last-axis encoding).
        os.split_axis = so.axis - so.full_rank;
        os.shard_dim_size = so.full_dim;
        os.producer_node = "state:" + so.variable_id;
        GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform output[" << so.output_index << "]: "
                       << tp_output_seam_to_string(os) << std::endl;
        output_seams.push_back(std::move(os));
    }

    GPU_DEBUG_INFO << "[GPU][TP] TPShardTransform detected " << seams.size() << " candidate node(s), sliced "
                   << sliced_nodes.size() << ", assembly outputs " << output_seams.size() << ", state seams "
                   << state_seams.size() << std::endl;

    set_tp_seams(model, std::move(seams));
    set_tp_output_seams(model, std::move(output_seams));
    set_tp_state_seams(model, std::move(state_seams));
    return graph_modified;
}

}  // namespace ov::intel_gpu
