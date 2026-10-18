// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04/S05/S07/S08/S09/S10/S11/S12/S13/S14] GATE tests for the GPU tensor-parallel compile + infer path.
// S04: Plugin::compile_model with model_distribution_policy{TENSOR_PARALLEL} over a 2-GPU
//      mini network (Parameter -> MatMul -> Result) must succeed and report k=2 ranks via
//      ov::intel_gpu::tp_ranks.
// S05: TPInferRequest executes the replicated graph on both ranks and exposes rank-0 outputs;
//      the result must match a single-GPU reference inference (e2e parity).
// S06: TPShardTransform (identity/analysis) detects MatMul/FullyConnected seams and reports
//      them via ov::intel_gpu::tp_seams; the graph/numbers are unchanged (S05 parity holds).
// S07: N-split - TPShardTransform slices the constant weights along the output-channel axis
//      so each rank uploads only its slice; rank 0 assembles the full output (all-gather
//      equivalent, host-side concatenation). Non-divisible N falls back to REPLICA.
// S08: K-split - TPShardTransform slices the constant weights AND the activation along the
//      contraction axis; each rank computes a partial GEMM sum; rank 0 all-reduces (fp32 sum)
//      and adds the replicated bias/residual terms exactly once.
// S09: int4_ASYM group quantized - TPShardTransform recognizes the dequant subgraph feeding the
//      GEMM B operand (weight u4 [N,G,gs] -> Convert -> Subtract(zp) -> Multiply(scale) ->
//      Reshape -> Convert) and slices weight + scale + zero_point on group boundaries (N-split:
//      whole out-channel chunks; K-split: whole groups, K multiple of group_size * num_ranks).
//      Indivisible/unsliceable cases fall back to REPLICA.
// S10: stateful recurrent state - TPShardTransform head-shards a head-local state variable
//      (ReadValue -> element-wise -> Assign) along the head axis, splitting it per rank and
//      aggregating it back to the full head shape on query_state(); non-head-local state stays
//      replicated (rank 0 authoritative). SDPA/GQA KV cache is head-sharded.
// S11: linear attention - a head-local state whose recurrence runs through a depthwise
//      GroupConvolution is head-sharded along the channel (head) axis with the depthwise filters
//      sliced per rank; chains that mix heads (real Qwen conv/ssm chains via Concat/Loop) stay
//      replicated.
// S12: dynamic state shapes - a partially dynamic state (dynamic batch, static divisible head
//      axis, the shape of every real Qwen3.5 state) is still head-sharded; the aggregated state
//      exposes the runtime full shape. Inter-rank collectives are routed through the host-USM
//      staging transport, and state query/set round-trips + concurrent (sync/async) requests match
//      single-GPU inference.
// S13: GenAI integration - the OpenVINO GenAI continuous-batching path forwards tensor parallelism
//      as a *string* config entry {"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}; a stateful model
//      compiled through that config and driven via the public name-based tensor API (as LLMPipeline
//      does) must reach the TP branch and round-trip its head-sharded state.
// S14: rank scheduler - the k rank sub-requests are dispatched in parallel on a shared k-stream
//      executor (one rank per device); several overlapping async requests plus a repeated
//      synchronous infer must still match single-GPU inference deterministically.
// S15: perf fast-paths for the collectives - a real `zeDeviceCanAccessPeer` probe selects the
//      PEER transport when available (host-USM staging otherwise, as on 2x A770), and the
//      all-reduce uses a ring reduce-scatter + all-gather (still fp32, replicated terms once).
//      The ring must stay deterministic across repeated inference and keep matching single-GPU.
// See PLAN.md §4.1, §4.2, §4.3, §4.4, §5, §7.4 (S04-S15).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

// A trivial single-FullyConnected-like model: Parameter -> MatMul -> Result.
std::shared_ptr<ov::Model> make_fc_mini_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, {1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_fc_mini");
}

// A projection whose (partial) output is added back to a residual stream: the heuristic in
// TPShardTransform must classify it as K_SPLIT (all-reduce(sum)) rather than N_SPLIT.
std::shared_ptr<ov::Model> make_residual_projection_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, {1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto add = std::make_shared<ov::op::v1::Add>(matmul, matmul);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_residual_proj");
}

// Same topology but with distinctive, non-uniform weights: parity mismatches caused by a rank
// not receiving the input (or the wrong rank's output being read) turn into large numeric deltas.
std::shared_ptr<ov::Model> make_tp_parity_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    std::vector<float> weights_data(8 * 16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        const float mag = 0.05f * static_cast<float>((i % 7) + 1);
        weights_data[i] = (i % 2 == 0) ? mag : -mag;
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_parity");
}

// [S07] An MLP-like column-parallel projection whose output terminates the model through a
// purely element-wise activation: Parameter -> MatMul (N-split) -> Relu -> Result. With N=16
// and 2 ranks each rank holds 8 output channels; rank 0 must expose the assembled full [1,16].
std::shared_ptr<ov::Model> make_mlp_nsplit_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    std::vector<float> weights_data(8 * 16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        const float mag = 0.05f * static_cast<float>((i % 5) + 1);
        weights_data[i] = (i % 2 == 0) ? mag : -mag;
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto relu = std::make_shared<ov::op::v0::Relu>(matmul);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_mlp_nsplit");
}

// [S07] Same topology but with an output-channel count (15) not divisible by the number of
// ranks: the pass must gracefully fall back to REPLICA (no slicing) instead of failing.
std::shared_ptr<ov::Model> make_indivisible_projection_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    std::vector<float> weights_data(8 * 15);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = 0.03f * static_cast<float>((i % 4) + 1);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 15}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_indivisible");
}

// [S08] Residual projection: Parameter(x) -> MatMul(W) -> Add(residual) -> Add(bias) -> Result.
// The MatMul output feeds an Add, so it is classified K_SPLIT. With tensor parallelism each
// rank computes a partial GEMM over its slice of the contraction axis; the replicated residual
// (a model input) and bias (a constant) must be added exactly once by the reducer.
std::shared_ptr<ov::Model> make_ksplit_residual_model() {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    std::vector<float> weights_data(8 * 16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        const float mag = 0.04f * static_cast<float>((i % 6) + 1);
        weights_data[i] = (i % 3 == 0) ? mag : -mag;
    }
    std::vector<float> bias_data(16);
    for (size_t i = 0; i < bias_data.size(); ++i) {
        bias_data[i] = 0.05f * static_cast<float>((i % 4) + 1) - 0.1f;
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, weights_data);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16}, bias_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, weights, false, false);
    auto add_residual = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto add_bias = std::make_shared<ov::op::v1::Add>(add_residual, bias);
    auto result = std::make_shared<ov::op::v0::Result>(add_bias);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{x, residual}, "tp_ksplit_residual");
}

// [S08] Same residual topology but with a contraction dimension (7) not divisible by the number
// of ranks: the pass must NOT slice (no all-reduce) and the replicated full computation must
// still match single-GPU inference.
std::shared_ptr<ov::Model> make_ksplit_indivisible_model() {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 7});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    std::vector<float> weights_data(7 * 16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = 0.03f * static_cast<float>((i % 5) + 1);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{7, 16}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, weights, false, false);
    auto add_residual = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto result = std::make_shared<ov::op::v0::Result>(add_residual);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{x, residual}, "tp_ksplit_indivisible");
}

// [S09] Builds the int4_ASYM group-quantized dequantization subgraph feeding the GEMM B operand:
//   weight(u4 [N, G, gs]) -> Convert(f16) -> Subtract(zp[u4->f16 N,G,1])
//                         -> Multiply(scale[f16 N,G,1]) -> Reshape([N, K]) -> Convert(f32)
// evaluated exactly (w = (q - zp) * scale) so the reference graph computes the true dequantized
// weights. Slicing weight/scale/zp on group boundaries must keep the result identical.
ov::Output<ov::Node> int4_asym_dequant_weights(size_t n, size_t k, size_t gs) {
    OPENVINO_ASSERT(k % gs == 0, "[TP] int4 test: K must be a multiple of group_size");
    const size_t g = k / gs;
    std::vector<int32_t> q(n * g * gs);
    std::vector<int32_t> zp(n * g);
    std::vector<ov::float16> scale(n * g);
    for (size_t nn = 0; nn < n; ++nn) {
        for (size_t gg = 0; gg < g; ++gg) {
            zp[nn * g + gg] = static_cast<int32_t>((nn * 3 + gg * 5) % 16);
            scale[nn * g + gg] = static_cast<ov::float16>(0.01f * static_cast<float>(1 + ((nn + gg) % 5)));
            for (size_t j = 0; j < gs; ++j) {
                q[(nn * g + gg) * gs + j] = static_cast<int32_t>((nn + gg + j) % 16);
            }
        }
    }
    auto w = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{n, g, gs}, q);
    auto zp_c = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{n, g, 1}, zp);
    auto sc_c = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, g, 1}, scale);
    auto w_f = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    auto zp_f = std::make_shared<ov::op::v0::Convert>(zp_c, ov::element::f16);
    auto sub = std::make_shared<ov::op::v1::Subtract>(w_f, zp_f);
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, sc_c);
    auto rp = ov::op::v0::Constant::create(ov::element::i64,
                                           ov::Shape{2},
                                           std::vector<int64_t>{static_cast<int64_t>(n), static_cast<int64_t>(k)});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(mul, rp, false);
    auto dq = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f32);
    return dq->output(0);
}

// [S09] N-split int4 model: Parameter -> MatMul(dequant int4 weights) -> Relu -> Result. The GEMM
// output terminates the model through an element-wise activation, so it is N-split (all-gather).
std::shared_ptr<ov::Model> make_int4_nsplit_model(size_t n = 4096, size_t k = 1024, size_t gs = 128) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, k});
    auto dequant = int4_asym_dequant_weights(n, k, gs);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, dequant, false, true);
    auto relu = std::make_shared<ov::op::v0::Relu>(matmul);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{x}, "tp_int4_nsplit");
}

// [S09] K-split int4 model: Parameter(x) -> MatMul(dequant int4) -> Add(residual) -> Add(bias)
// -> Result. The projection feeds a residual Add, so it is classified K_SPLIT (all-reduce sum);
// the replicated residual (input) and bias (constant) must be added exactly once by the reducer.
std::shared_ptr<ov::Model> make_int4_ksplit_model(size_t n = 4096, size_t k = 1024, size_t gs = 128) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, k});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, n});
    auto dequant = int4_asym_dequant_weights(n, k, gs);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, dequant, false, true);
    std::vector<float> bias_data(n);
    for (size_t i = 0; i < n; ++i) {
        bias_data[i] = 0.02f * static_cast<float>((i % 7) + 1) - 0.07f;
    }
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{n}, bias_data);
    auto add_residual = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto add_bias = std::make_shared<ov::op::v1::Add>(add_residual, bias);
    auto result = std::make_shared<ov::op::v0::Result>(add_bias);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{x, residual}, "tp_int4_ksplit");
}

// [S09] Negative: the number of groups (G = K/gs) is not divisible by the number of ranks, so the
// contraction axis cannot be sliced on group boundaries -> the node must fall back to replication
// (K_SPLIT classification preserved, no all-reduce output) and still match single-GPU inference.
std::shared_ptr<ov::Model> make_int4_ksplit_indivisible_model(size_t n = 4096, size_t k = 1152, size_t gs = 128) {
    return make_int4_ksplit_model(n, k, gs);  // k/gs = 9 groups (odd)
}

// [S10] Stateful model with a head-shardable recurrent state:
//   Variable V[1, H, hd] -> ReadValue -> Add(x) -> Assign(V) ; Result(Add)
// When @p broadcast_input is true, x is [1, 1, hd] so it broadcasts along the head axis and the
// whole ReadValue -> Add -> Assign chain is head-local: the pass must head-shard V (heads split
// across ranks, aggregated on query_state()). Otherwise x is full-width [1, H, hd] and the pass
// must keep V replicated.
std::shared_ptr<ov::Model> make_stateful_head_model(size_t heads, size_t head_dim, bool broadcast_input) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1, static_cast<int64_t>(heads), static_cast<int64_t>(head_dim)},
                                   ov::element::f32,
                                   "tp_state_var"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    const ov::Shape x_shape = broadcast_input ? ov::Shape{1, 1, head_dim} : ov::Shape{1, heads, head_dim};
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
    auto add = std::make_shared<ov::op::v1::Add>(read_value, x);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       ov::ParameterVector{x},
                                       "tp_state_head");
}

// [S11] A minimal linear-attention-like block: a per-head state updated through a depthwise
// GroupConvolution (the PLAN §4.3 "depthwise conv per head" primitive). The variable holds
// [1, heads, taps]; ReadValue -> depthwise GroupConvolution (per-channel kernel, same padding so
// the output keeps [1, heads, taps]) -> Assign. Every output channel depends only on the matching
// input channel, so the chain is head-local: the pass must head-shard the variable along the head
// axis (axis 1) and slice the depthwise filters (GROUPS = heads) by the same rank range.
std::shared_ptr<ov::Model> make_linear_attn_conv_state_model(size_t heads, size_t taps) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1, static_cast<int64_t>(heads), static_cast<int64_t>(taps)},
                                   ov::element::f32,
                                   "tp_la_conv_state"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    // Depthwise filters [GROUPS=heads, C_OUT/GROUPS=1, C_IN/GROUPS=1, k=taps].
    std::vector<float> filter_data(heads * taps);
    for (size_t i = 0; i < filter_data.size(); ++i) {
        filter_data[i] = 0.2f * static_cast<float>((i % 5) + 1) - 0.5f;
    }
    auto filters = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{heads, 1, 1, taps}, filter_data);
    const ov::Strides strides{1};
    const ov::CoordinateDiff pads_begin{static_cast<std::ptrdiff_t>(taps / 2)};
    const ov::CoordinateDiff pads_end{static_cast<std::ptrdiff_t>(taps - 1 - taps / 2)};
    const ov::Strides dilations{1};
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(read_value,
                                                               filters,
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               dilations,
                                                               ov::op::PadType::EXPLICIT);
    auto assign = std::make_shared<ov::op::v6::Assign>(conv, variable);
    auto result = std::make_shared<ov::op::v0::Result>(conv);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       ov::ParameterVector{},
                                       "tp_la_conv_state");
}

// [S12] Partially dynamic, head-shardable recurrent state:
//   Variable V[?, H, hd] -> ReadValue -> Add(x[?, 1, hd]) -> Assign(V) ; Result(Add)
// The variable has a dynamic batch axis but a static head axis (axis 1) divisible by the number
// of ranks, mirroring the real Qwen3.5 states (`?,4,?,256` KV, `?,8192,4` conv, `?,32,128,128`
// ssm). The ReadValue -> Add(broadcast) -> Assign chain is head-local, so the variable must be
// HEAD_SHARDED despite the dynamic dimension; a head count not divisible by the ranks must fall
// back to REPLICATED.
std::shared_ptr<ov::Model> make_dynamic_stateful_head_model(size_t heads, size_t head_dim) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{ov::Dimension::dynamic(),
                                                    static_cast<int64_t>(heads),
                                                    static_cast<int64_t>(head_dim)},
                                   ov::element::f32,
                                   "tp_dyn_state_var"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    auto x = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), 1, static_cast<int64_t>(head_dim)});
    auto add = std::make_shared<ov::op::v1::Add>(read_value, x);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       ov::ParameterVector{x},
                                       "tp_dyn_state_head");
}

// [S13] GenAI-style stateful block with named IO and a dynamic sequence axis. This mirrors how
// OpenVINO GenAI's LLMPipeline / continuous-batching pipeline drives a stateful decoder: it binds
// tensors by *name* (set_tensor("input_ids", ...)/get_tensor("logits")) and manipulates state
// through query_state()/set_state(). The ReadValue -> Add(broadcast) -> Assign chain is head-local
// with a static divisible head axis, so it must be HEAD_SHARDED (aggregated at the GenAI boundary).
std::shared_ptr<ov::Model> make_genai_style_stateful_model(size_t heads, size_t head_dim) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{ov::Dimension::dynamic(),
                                                    static_cast<int64_t>(heads),
                                                    static_cast<int64_t>(head_dim)},
                                   ov::element::f32,
                                   "past_key_values.0"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
    auto x = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), 1, static_cast<int64_t>(head_dim)});
    x->set_friendly_name("input_ids");
    x->get_output_tensor(0).add_names({"input_ids"});
    auto add = std::make_shared<ov::op::v1::Add>(read_value, x);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("logits");
    result->get_output_tensor(0).add_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       ov::ParameterVector{x},
                                       "tp_genai_stateful");
}

// Flattens a tensor to fp32 for numeric comparison (handles the GPU default f16 outputs).
std::vector<float> to_floats(const ov::Tensor& tensor) {
    const auto count = tensor.get_size();
    std::vector<float> out(count);
    const auto et = tensor.get_element_type();
    if (et == ov::element::f16) {
        const auto* src = tensor.data<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
    } else if (et == ov::element::bf16) {
        const auto* src = tensor.data<ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
    } else {
        const auto* src = tensor.data<float>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = src[i];
        }
    }
    return out;
}

// Cosine similarity between two equally-sized tensors.
double cosine_similarity(const ov::Tensor& a, const ov::Tensor& b) {
    OPENVINO_ASSERT(a.get_size() == b.get_size(), "[TP] cosine_similarity size mismatch");
    const auto va = to_floats(a);
    const auto vb = to_floats(b);
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < va.size(); ++i) {
        dot += static_cast<double>(va[i]) * vb[i];
        na += static_cast<double>(va[i]) * va[i];
        nb += static_cast<double>(vb[i]) * vb[i];
    }
    if (na == 0.0 || nb == 0.0) {
        return (na == nb) ? 1.0 : 0.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Number of individually addressable GPU devices (excludes sub-devices such as "GPU.0.0").
size_t count_gpu_devices(ov::Core& core) {
    size_t count = 0;
    for (const auto& device : core.get_available_devices()) {
        // "GPU.N" - exactly one dot, no further sub-device suffix.
        if (device.rfind("GPU.", 0) == 0 && device.find('.', 4) == std::string::npos) {
            ++count;
        }
    }
    return count;
}

class TensorParallelCompileGPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = make_fc_mini_model();
    }
};

TEST_F(TensorParallelCompileGPU, CompilesMiniModelOnTwoGpus) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props));
    ASSERT_NE(compiled_model, nullptr);

    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    ASSERT_EQ(ranks.size(), 2u) << "Expected exactly 2 tensor-parallel ranks";
    EXPECT_EQ(ranks, (std::vector<int32_t>{0, 1}));
}

TEST_F(TensorParallelCompileGPU, InternalFlagAlsoEnablesTensorParallel) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    ov::AnyMap props = {ov::intel_gpu::enable_tensor_parallel(true)};

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props));

    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_EQ(ranks.size(), 2u);
}

TEST_F(TensorParallelCompileGPU, DefaultCompileHasNoRanks) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU));
    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_TRUE(ranks.empty());
}

TEST_F(TensorParallelCompileGPU, RejectsExplicitDeviceId) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    // Tensor parallelism spans all devices, so "GPU.0" is ambiguous and must be rejected.
    EXPECT_THROW(core.compile_model(shared_model, "GPU.0", props), ov::Exception);
}

TEST_F(TensorParallelCompileGPU, RejectsPipelineParallel) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> pp_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(pp_policy)};

    EXPECT_THROW(core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props), ov::Exception);
}

// [S05] GATE: tensor-parallel inference (2 GPUs) must match single-GPU inference on the same
// replicated mini model. Ranks are identical at this stage, so no sharding arithmetic is involved.
TEST_F(TensorParallelCompileGPU, TensorParallelInferMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_tp_parity_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) - 3.5f;
        }
    }

    // Reference: plain single-GPU inference.
    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(input);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    // Tensor parallel: replicated graph on both GPUs, outputs from rank 0.
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});
    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(input);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
}

// [S06] GATE: TPShardTransform must detect the MatMul seam and report it via tp_seams.
// The list must be non-empty, classified and stable across repeated compilations.
TEST_F(TensorParallelCompileGPU, ShardTransformDetectsAndReportsSeams) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    auto compiled = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props);
    auto seams = compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u) << "Expected exactly one detected MatMul seam";
    // Format: "<kind>:<collective>:<node_type>:<node_name>".
    EXPECT_EQ(seams[0].rfind("N_SPLIT:ALL_GATHER:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];

    // Determinism/stability: recompiling the same model yields an identical seam list.
    auto compiled_again = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props);
    auto seams_again = compiled_again.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    EXPECT_EQ(seams, seams_again);

    // The seam list must be advertised among the supported properties.
    auto supported = compiled.get_property(ov::supported_properties).as<std::vector<ov::PropertyName>>();
    EXPECT_NE(std::find(supported.begin(), supported.end(), ov::intel_gpu::tp_seams.name()), supported.end());
}

// [S06] Classification: a projection whose output is added back to the residual stream is
// classified as K_SPLIT (all-reduce(sum)); the mini model above stays N_SPLIT (all-gather).
TEST_F(TensorParallelCompileGPU, ShardTransformClassifiesResidualProjectionAsKSplit) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto residual_model = make_residual_projection_model();
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    auto compiled = core.compile_model(residual_model, ov::test::utils::DEVICE_GPU, props);
    auto seams = compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u) << "Expected exactly one detected MatMul seam";
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
}

// [S06] Regression: TP-off compilation must not run the pass and must report no seams/ranks.
TEST_F(TensorParallelCompileGPU, DefaultCompileHasNoSeams) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU));
    auto seams = compiled_model.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    EXPECT_TRUE(seams.empty());
    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_TRUE(ranks.empty());
}

// [S07] GATE: N-split weight slicing + all-gather equivalent. Each rank holds only its
// output-channel slice of the projection weights; rank 0 exposes the full [1,16] tensor
// assembled from the per-rank slices. The result must match single-GPU inference.
TEST_F(TensorParallelCompileGPU, NsplitMlpParityMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_mlp_nsplit_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) - 3.5f;
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(input);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    // The N-split seam is detected; rank-0 output is advertised at the full N shape.
    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("N_SPLIT:ALL_GATHER:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(input);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    // Shape contract: rank 0 must expose the FULL tensor, not its slice.
    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ASSERT_EQ(tp_output.get_shape(), (ov::Shape{1, 16}));

    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999)
        << "N-split all-gather output diverged from the single-GPU reference";
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S07] Negative: an output-channel count not divisible by the number of ranks must not fail
// compilation - the pass degrades the seam to REPLICA (full weights on every rank) and the
// replicated result still matches single-GPU inference.
TEST_F(TensorParallelCompileGPU, NsplitNotDivisibleFallsBackToReplica) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_indivisible_projection_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) * 0.5f - 1.0f;
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(input);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::CompiledModel tp_compiled;
    OV_ASSERT_NO_THROW(tp_compiled = core.compile_model(
                          tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)}));

    // Graceful fallback: the seam is reported as REPLICA (not sliced/assembled).
    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("REPLICA:", 0), 0u) << "Unexpected seam: " << seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(input);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999);
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S08] GATE: K-split + all-reduce(sum). The residual projection is classified K_SPLIT; each
// rank computes a partial GEMM on its contraction-axis slice and rank 0 exposes the full
// result after summing the partials in fp32 and adding the replicated residual/bias once.
TEST_F(TensorParallelCompileGPU, KsplitResidualParityMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_ksplit_residual_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) * 0.5f - 2.0f;
        }
    }
    ov::Tensor residual(ov::element::f32, ov::Shape{1, 16});
    {
        auto* data = residual.data<float>();
        for (size_t i = 0; i < 16; ++i) {
            data[i] = 0.25f * static_cast<float>((i % 5) + 1) - 0.6f;
        }
    }

    // Reference: plain single-GPU inference.
    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(0, input);
    ref_request.set_input_tensor(1, residual);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    // The projection must be classified K_SPLIT and set up as an all-reduce(sum) output.
    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    ASSERT_EQ(output_seams.size(), 1u) << "Expected exactly one all-reduce output seam";
    EXPECT_EQ(output_seams[0].rfind("ALL_REDUCE_SUM:", 0), 0u) << "Unexpected output seam: " << output_seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(0, input);
    tp_request.set_input_tensor(1, residual);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ASSERT_EQ(tp_output.get_shape(), (ov::Shape{1, 16}));
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999)
        << "K-split all-reduce output diverged from the single-GPU reference";
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S08] Negative: a contraction dimension not divisible by the number of ranks must not be
// sliced. The seam stays K_SPLIT (stable classification) but no all-reduce output is set up and
// the replicated full computation still matches single-GPU inference.
TEST_F(TensorParallelCompileGPU, KsplitNotDivisibleFallsBackToReplica) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_ksplit_indivisible_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 7});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 7; ++i) {
            data[i] = static_cast<float>(i) - 3.0f;
        }
    }
    ov::Tensor residual(ov::element::f32, ov::Shape{1, 16});
    {
        auto* data = residual.data<float>();
        for (size_t i = 0; i < 16; ++i) {
            data[i] = 0.1f * static_cast<float>(i + 1);
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(0, input);
    ref_request.set_input_tensor(1, residual);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    // Classification is stable (no hard failure) ...
    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
    // ... but nothing was sliced, so there is no assembly output (rank 0 computes the full result).
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    EXPECT_TRUE(output_seams.empty());

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(0, input);
    tp_request.set_input_tensor(1, residual);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999);
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S15] The ring all-reduce must be deterministic: repeated inference on the same request and a
// fresh request must produce bit-identical full-shape outputs (the ring's fixed chunk order),
// while still matching the single-GPU reference (fp32 accumulation, replicated term once).
TEST_F(TensorParallelCompileGPU, KsplitRingAllReduceDeterministicAcrossRuns) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_ksplit_residual_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) * 0.25f - 1.0f;
        }
    }
    ov::Tensor residual(ov::element::f32, ov::Shape{1, 16});
    {
        auto* data = residual.data<float>();
        for (size_t i = 0; i < 16; ++i) {
            data[i] = 0.125f * static_cast<float>((i % 3) + 1);
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(0, input);
    ref_request.set_input_tensor(1, residual);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    // This model is classified K_SPLIT with an all-reduce output seam (ring all-reduce path).
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    ASSERT_EQ(output_seams.size(), 1u) << "Expected exactly one all-reduce output seam";
    EXPECT_EQ(output_seams[0].rfind("ALL_REDUCE_SUM:", 0), 0u) << "Unexpected output seam: " << output_seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(0, input);
    tp_request.set_input_tensor(1, residual);
    tp_request.infer();
    const ov::Tensor first = tp_request.get_output_tensor();
    std::vector<float> first_data(first.get_size());
    std::copy_n(first.data<float>(), first.get_size(), first_data.begin());

    // Repeated inference on the same request must be bit-identical (deterministic ring order).
    for (int run = 0; run < 3; ++run) {
        tp_request.infer();
        const ov::Tensor again = tp_request.get_output_tensor();
        ASSERT_EQ(again.get_size(), first_data.size());
        for (size_t i = 0; i < first_data.size(); ++i) {
            EXPECT_FLOAT_EQ(first_data[i], again.data<float>()[i])
                << "Ring all-reduce is not deterministic at index " << i << " (run " << run << ")";
        }
    }

    // A fresh request must also be bit-identical and match single-GPU inference.
    auto fresh_request = tp_compiled.create_infer_request();
    fresh_request.set_input_tensor(0, input);
    fresh_request.set_input_tensor(1, residual);
    fresh_request.infer();
    const ov::Tensor fresh = fresh_request.get_output_tensor();
    ASSERT_EQ(fresh.get_shape(), ref_output.get_shape());
    EXPECT_GT(cosine_similarity(ref_output, fresh), 0.999);
    ov::test::utils::compare(ref_output, fresh, 1e-3f, 1e-3f);
    for (size_t i = 0; i < first_data.size(); ++i) {
        EXPECT_FLOAT_EQ(first_data[i], fresh.data<float>()[i]) << "Fresh request diverged at index " << i;
    }
}

// [S09] GATE: N-split of an int4_ASYM group-quantized projection. Each rank holds only its
// out-channel chunk of weight + scale + zero_point (whole groups on the K axis are preserved);
// rank 0 exposes the full [1, N] result assembled by all-gather, matching single-GPU inference.
TEST_F(TensorParallelCompileGPU, Int4NsplitParityMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t n = 4096;
    const size_t k = 1024;
    auto ref_model = make_int4_nsplit_model(n, k);
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, k});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < k; ++i) {
            data[i] = 0.05f * static_cast<float>((i % 11) + 1) - 0.3f;
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(input);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("N_SPLIT:ALL_GATHER:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    ASSERT_EQ(output_seams.size(), 1u) << "Expected the int4 N-split to set up an all-gather output";
    EXPECT_EQ(output_seams[0].rfind("ALL_GATHER:", 0), 0u) << "Unexpected output seam: " << output_seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(input);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ASSERT_EQ(tp_output.get_shape(), (ov::Shape{1, n}));
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999)
        << "int4 N-split all-gather output diverged from the single-GPU reference";
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S09] GATE: K-split of an int4_ASYM projection. Each rank keeps its whole-group slice of
// weight + scale + zero_point (G divisible by the number of ranks) and computes a partial GEMM;
// rank 0 all-reduces (fp32 sum) and adds the replicated residual/bias exactly once.
TEST_F(TensorParallelCompileGPU, Int4KsplitParityMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t n = 4096;
    const size_t k = 1024;
    auto ref_model = make_int4_ksplit_model(n, k);
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, k});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < k; ++i) {
            data[i] = 0.05f * static_cast<float>((i % 9) + 1) - 0.25f;
        }
    }
    ov::Tensor residual(ov::element::f32, ov::Shape{1, n});
    {
        auto* data = residual.data<float>();
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.02f * static_cast<float>((i % 13) + 1) - 0.1f;
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(0, input);
    ref_request.set_input_tensor(1, residual);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    ASSERT_EQ(output_seams.size(), 1u) << "Expected the int4 K-split to set up an all-reduce output";
    EXPECT_EQ(output_seams[0].rfind("ALL_REDUCE_SUM:", 0), 0u) << "Unexpected output seam: " << output_seams[0];

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(0, input);
    tp_request.set_input_tensor(1, residual);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ASSERT_EQ(tp_output.get_shape(), (ov::Shape{1, n}));
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999)
        << "int4 K-split all-reduce output diverged from the single-GPU reference";
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S09] Negative: the number of quantization groups (G = K / group_size) is not divisible by the
// number of ranks, so the contraction axis cannot be split on group boundaries. The seam stays
// K_SPLIT (stable classification) but is not sliced (no all-reduce output) and the replicated
// full computation must still match single-GPU inference.
TEST_F(TensorParallelCompileGPU, Int4KsplitIndivisibleFallsBackToReplica) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t n = 4096;
    const size_t k = 1152;  // k / 128 = 9 groups (odd, not divisible by 2 ranks)
    auto ref_model = make_int4_ksplit_indivisible_model(n, k);
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, k});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < k; ++i) {
            data[i] = 0.05f * static_cast<float>((i % 7) + 1) - 0.2f;
        }
    }
    ov::Tensor residual(ov::element::f32, ov::Shape{1, n});
    {
        auto* data = residual.data<float>();
        for (size_t i = 0; i < n; ++i) {
            data[i] = 0.03f * static_cast<float>((i % 5) + 1) - 0.1f;
        }
    }

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(0, input);
    ref_request.set_input_tensor(1, residual);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});

    auto seams = tp_compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u);
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
    auto output_seams = tp_compiled.get_property(ov::intel_gpu::tp_output_seams).as<std::vector<std::string>>();
    EXPECT_TRUE(output_seams.empty()) << "Indivisible group count must not set up an all-reduce output";

    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(0, input);
    tp_request.set_input_tensor(1, residual);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    EXPECT_GT(cosine_similarity(ref_output, tp_output), 0.999);
    ov::test::utils::compare(ref_output, tp_output, 1e-3f, 1e-3f);
}

// [S10] GATE: a head-local static state variable must be head-sharded - reported via
// tp_state_seams, split per rank (each rank keeping H/num_ranks heads) and aggregated back to the
// full state on query_state(). Multi-step TP inference must match single-GPU over the sequence.
TEST_F(TensorParallelCompileGPU, StateHeadShardingMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    auto ref_model = make_stateful_head_model(heads, head_dim, /*broadcast_input=*/true);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    // "HEAD_SHARDED:<axis>:<full_dim>:<variable_id>" -> axis 1, full_dim 4.
    EXPECT_EQ(state_seams[0].rfind("HEAD_SHARDED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];
    EXPECT_NE(state_seams[0].find(":1:4:"), std::string::npos) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    const size_t steps = 5;
    for (size_t t = 0; t < steps; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{1, 1, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < head_dim; ++i) {
            data[i] = 0.25f * static_cast<float>(t + 1) + static_cast<float>(i);
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    // The aggregated TP state must expose the FULL head shape (not the per-rank slice).
    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    const ov::Tensor tp_state = tp_states[0].get_state();
    EXPECT_EQ(tp_state.get_shape(), (ov::Shape{1, heads, head_dim}));
    auto ref_states = ref_request.query_state();
    ASSERT_EQ(ref_states.size(), 1u);
    const ov::Tensor ref_state = ref_states[0].get_state();
    ASSERT_EQ(ref_state.get_shape(), tp_state.get_shape());
    ov::test::utils::compare(ref_state, tp_state, 1e-4f, 1e-4f);
}

// [S10] Negative: a full-width state operand makes the ReadValue chain non-head-local, so the
// variable must stay REPLICATED (rank 0 authoritative); TP inference must still match single-GPU.
TEST_F(TensorParallelCompileGPU, StateNonBroadcastOperandStaysReplicated) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    auto ref_model = make_stateful_head_model(heads, head_dim, /*broadcast_input=*/false);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    EXPECT_EQ(state_seams[0].rfind("REPLICATED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    const size_t steps = 4;
    for (size_t t = 0; t < steps; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{1, heads, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < heads * head_dim; ++i) {
            data[i] = 0.1f * static_cast<float>((i + t) % 7) - 0.3f;
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    // Replicated state is still exposed with the full shape on rank 0.
    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    EXPECT_EQ(tp_states[0].get_state().get_shape(), (ov::Shape{1, heads, head_dim}));
}

// [S10] A plain (single-GPU) compilation must not report any tensor-parallel state seams.
TEST_F(TensorParallelCompileGPU, DefaultCompileHasNoStateSeams) {
    ov::Core core;
    auto plain_model = make_stateful_head_model(4, 2, /*broadcast_input=*/true);
    auto compiled = core.compile_model(plain_model, ov::test::utils::DEVICE_GPU);
    auto state_seams = compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    EXPECT_TRUE(state_seams.empty());
}

// [S11] GATE: a head-local linear-attention state whose recurrence runs through a depthwise
// GroupConvolution must be head-sharded (axis 1 = heads) with the depthwise filters sliced per
// rank. Multi-step TP inference must match single-GPU over the recurrence and the aggregated
// state must expose the full head shape on query_state().
TEST_F(TensorParallelCompileGPU, LinearAttentionConvStateHeadShardingMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t taps = 3;
    auto ref_model = make_linear_attn_conv_state_model(heads, taps);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    // "HEAD_SHARDED:<axis>:<full_dim>:<variable_id>" -> axis 1, full_dim 4.
    EXPECT_EQ(state_seams[0].rfind("HEAD_SHARDED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];
    EXPECT_NE(state_seams[0].find(":1:4:"), std::string::npos) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    // Seed a non-zero, per-head-distinct initial state so the conv recurrence is actually
    // exercised and the rank<->head scatter is checked (HEAD_SHARDED path).
    ov::Tensor init_state(ov::element::f32, ov::Shape{1, heads, taps});
    auto* init_data = init_state.data<float>();
    for (size_t i = 0; i < heads * taps; ++i) {
        init_data[i] = 0.25f * static_cast<float>(i + 1) - 0.5f;
    }
    ref_request.query_state()[0].set_state(init_state);
    tp_request.query_state()[0].set_state(init_state);

    const size_t steps = 4;
    for (size_t t = 0; t < steps; ++t) {
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    // The aggregated TP state must expose the FULL head shape (not the per-rank slice).
    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    const ov::Tensor tp_state = tp_states[0].get_state();
    EXPECT_EQ(tp_state.get_shape(), (ov::Shape{1, heads, taps}));
    auto ref_states = ref_request.query_state();
    ASSERT_EQ(ref_states.size(), 1u);
    ASSERT_EQ(ref_states[0].get_state().get_shape(), tp_state.get_shape());
    ov::test::utils::compare(ref_states[0].get_state(), tp_state, 1e-4f, 1e-4f);
}

// [S11] Negative: a head count not divisible by the number of ranks makes the depthwise conv state
// non-head-local, so the variable must stay REPLICATED (no filter slicing, rank 0 authoritative)
// and TP inference must still match single-GPU over the recurrence.
TEST_F(TensorParallelCompileGPU, LinearAttentionConvStateIndivisibleStaysReplicated) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 3;  // not divisible by 2 ranks
    const size_t taps = 3;
    auto ref_model = make_linear_attn_conv_state_model(heads, taps);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    EXPECT_EQ(state_seams[0].rfind("REPLICATED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    // A non-zero initial state (rank-0 fan-out path for REPLICATED) is set on both requests.
    ov::Tensor init_state(ov::element::f32, ov::Shape{1, heads, taps});
    auto* init_data = init_state.data<float>();
    for (size_t i = 0; i < heads * taps; ++i) {
        init_data[i] = 0.25f * static_cast<float>(i + 1) - 0.5f;
    }
    ref_request.query_state()[0].set_state(init_state);
    tp_request.query_state()[0].set_state(init_state);

    for (size_t t = 0; t < 3; ++t) {
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    EXPECT_EQ(tp_states[0].get_state().get_shape(), (ov::Shape{1, heads, taps}));
}

// [S12] GATE: a partially dynamic state variable (dynamic batch, static divisible head axis - the
// shape of every real Qwen3.5 state: `?,4,?,256` etc.) must be HEAD_SHARDED along the static head
// axis. Multi-step TP inference with a concrete dynamic batch must match single-GPU inference and
// the aggregated (full-shape) state must match too.
TEST_F(TensorParallelCompileGPU, DynamicStateHeadShardingMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    const size_t batch = 2;
    auto ref_model = make_dynamic_stateful_head_model(heads, head_dim);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    // The dynamic batch must not force replication: head axis 1 (size 4) is static and divisible.
    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    EXPECT_EQ(state_seams[0].rfind("HEAD_SHARDED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];
    EXPECT_NE(state_seams[0].find(":1:4:"), std::string::npos) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    for (size_t t = 0; t < 4; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{batch, 1, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < batch * head_dim; ++i) {
            data[i] = 0.2f * static_cast<float>((i + t) % 6) - 0.4f;
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    // Aggregated state exposes the runtime full shape (batch preserved, head axis restored).
    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    EXPECT_EQ(tp_states[0].get_state().get_shape(), (ov::Shape{batch, heads, head_dim}));
    auto ref_states = ref_request.query_state();
    ASSERT_EQ(ref_states.size(), 1u);
    ov::test::utils::compare(ref_states[0].get_state(), tp_states[0].get_state(), 1e-4f, 1e-4f);
}

// [S12] Negative: a dynamic state whose head axis is not divisible by the number of ranks must
// stay REPLICATED (rank 0 authoritative) and still match single-GPU inference.
TEST_F(TensorParallelCompileGPU, DynamicStateIndivisibleStaysReplicated) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 3;  // not divisible by 2 ranks
    const size_t head_dim = 2;
    auto ref_model = make_dynamic_stateful_head_model(heads, head_dim);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    EXPECT_EQ(state_seams[0].rfind("REPLICATED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    for (size_t t = 0; t < 3; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{1, 1, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < head_dim; ++i) {
            data[i] = 0.3f * static_cast<float>(t + 1) + static_cast<float>(i);
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_output_tensor();
        const ov::Tensor tp_output = tp_request.get_output_tensor();
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    EXPECT_EQ(tp_states[0].get_state().get_shape(), (ov::Shape{1, heads, head_dim}));
}

// [S12] State roundtrip: query_state() -> set_state() -> infer() for a HEAD_SHARDED variable must
// distribute the full state across ranks (via the staging transport) so a resumed TP request
// matches a single-GPU request resumed from the same state.
TEST_F(TensorParallelCompileGPU, StateRoundtripHeadShardedMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    auto ref_model = make_stateful_head_model(heads, head_dim, /*broadcast_input=*/true);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    // Reference resumed from a known full state, and TP resumed by scattering the same state.
    ov::Tensor seed(ov::element::f32, ov::Shape{1, heads, head_dim});
    auto* seed_data = seed.data<float>();
    for (size_t i = 0; i < heads * head_dim; ++i) {
        seed_data[i] = 0.2f * static_cast<float>(i + 1) - 0.35f;
    }

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();
    ref_request.query_state()[0].set_state(seed);
    tp_request.query_state()[0].set_state(seed);

    // Roundtrip through query_state(): capture the aggregated TP state and re-apply it.
    const ov::Tensor captured = tp_request.query_state()[0].get_state();
    ASSERT_EQ(captured.get_shape(), (ov::Shape{1, heads, head_dim}));
    tp_request.query_state()[0].set_state(captured);

    for (size_t t = 0; t < 3; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{1, 1, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < head_dim; ++i) {
            data[i] = 0.1f * static_cast<float>(t + 1) - static_cast<float>(i);
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        ov::test::utils::compare(ref_request.get_output_tensor(), tp_request.get_output_tensor(), 1e-4f, 1e-4f);
    }
}

// [S12] State roundtrip for a REPLICATED variable: set_state() fans the full state out to every
// rank and a resumed TP request matches single-GPU inference.
TEST_F(TensorParallelCompileGPU, StateRoundtripReplicatedMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    auto ref_model = make_stateful_head_model(heads, head_dim, /*broadcast_input=*/false);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    ov::Tensor seed(ov::element::f32, ov::Shape{1, heads, head_dim});
    auto* seed_data = seed.data<float>();
    for (size_t i = 0; i < heads * head_dim; ++i) {
        seed_data[i] = 0.15f * static_cast<float>(i + 1) - 0.25f;
    }

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();
    ref_request.query_state()[0].set_state(seed);
    tp_request.query_state()[0].set_state(seed);

    for (size_t t = 0; t < 3; ++t) {
        ov::Tensor x(ov::element::f32, ov::Shape{1, heads, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < heads * head_dim; ++i) {
            data[i] = 0.1f * static_cast<float>((i + t) % 7) - 0.3f;
        }
        ref_request.set_input_tensor(x);
        tp_request.set_input_tensor(x);
        ref_request.infer();
        tp_request.infer();
        ov::test::utils::compare(ref_request.get_output_tensor(), tp_request.get_output_tensor(), 1e-4f, 1e-4f);
    }
}

// [S12] Concurrency: several independent TP requests (sync and async) with different inputs must
// each match their single-GPU reference. This exercises the per-request isolation of the input
// replication / output assembly / transport buffers under the async executor.
TEST_F(TensorParallelCompileGPU, ConcurrentRequestsMatchSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_mlp_nsplit_model();
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    const size_t k = 2;
    std::vector<ov::Tensor> inputs;
    std::vector<ov::Tensor> ref_outputs;
    for (size_t n = 0; n < k; ++n) {
        ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = 0.1f * static_cast<float>(i + 1) * static_cast<float>(n + 1) - 0.5f;
        }
        auto ref_request = ref_compiled.create_infer_request();
        ref_request.set_input_tensor(input);
        ref_request.infer();
        inputs.push_back(input);
        ref_outputs.push_back(ref_request.get_output_tensor());
    }

    // Two independent synchronous requests.
    {
        auto r0 = tp_compiled.create_infer_request();
        auto r1 = tp_compiled.create_infer_request();
        r0.set_input_tensor(inputs[0]);
        r1.set_input_tensor(inputs[1]);
        r0.infer();
        r1.infer();
        ov::test::utils::compare(ref_outputs[0], r0.get_output_tensor(), 1e-3f, 1e-3f);
        ov::test::utils::compare(ref_outputs[1], r1.get_output_tensor(), 1e-3f, 1e-3f);
    }

    // Two overlapping asynchronous requests.
    {
        auto r0 = tp_compiled.create_infer_request();
        auto r1 = tp_compiled.create_infer_request();
        r0.set_input_tensor(inputs[0]);
        r1.set_input_tensor(inputs[1]);
        r0.start_async();
        r1.start_async();
        r0.wait();
        r1.wait();
        ov::test::utils::compare(ref_outputs[0], r0.get_output_tensor(), 1e-3f, 1e-3f);
        ov::test::utils::compare(ref_outputs[1], r1.get_output_tensor(), 1e-3f, 1e-3f);
    }
}

// [S14] Scheduler: the k rank sub-requests must execute in parallel on the compiled model's
// k-stream rank scheduler while the assembled outputs stay deterministic. This test fires several
// overlapping asynchronous requests (more than the number of ranks) on an N-split (all-gather)
// model and checks every result against its single-GPU reference; then it re-runs a single request
// twice and requires a bit-identical output, proving the parallel fan-out does not race the shared
// input-replication / staging paths.
TEST_F(TensorParallelCompileGPU, SchedulerParallelRanksMatchSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_mlp_nsplit_model();
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU,
                                          {ov::hint::model_distribution_policy(tp_policy)});

    const size_t num_requests = 4;  // 2x the number of ranks: forces rank-stream reuse.
    std::vector<ov::Tensor> inputs;
    std::vector<ov::Tensor> ref_outputs;
    inputs.reserve(num_requests);
    ref_outputs.reserve(num_requests);
    for (size_t n = 0; n < num_requests; ++n) {
        ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = 0.13f * static_cast<float>(i + 1) * static_cast<float>(n + 1) - 0.7f;
        }
        auto ref_request = ref_compiled.create_infer_request();
        ref_request.set_input_tensor(input);
        ref_request.infer();
        inputs.push_back(input);
        ref_outputs.push_back(ref_request.get_output_tensor());
    }

    // All requests start before any is waited on: the rank scheduler is exercised concurrently.
    std::vector<ov::InferRequest> tp_requests;
    tp_requests.reserve(num_requests);
    for (size_t n = 0; n < num_requests; ++n) {
        auto request = tp_compiled.create_infer_request();
        request.set_input_tensor(inputs[n]);
        request.start_async();
        tp_requests.push_back(request);
    }
    for (size_t n = 0; n < num_requests; ++n) {
        tp_requests[n].wait();
        ov::test::utils::compare(ref_outputs[n], tp_requests[n].get_output_tensor(), 1e-3f, 1e-3f);
    }

    // Determinism under a repeated (synchronous) infer on one request.
    auto stable_request = tp_compiled.create_infer_request();
    stable_request.set_input_tensor(inputs[0]);
    stable_request.infer();
    const ov::Tensor first = stable_request.get_output_tensor();
    std::vector<float> first_values(first.get_size());
    ov::test::utils::compare(ref_outputs[0], first, 1e-3f, 1e-3f);
    {
        const auto count = first.get_size();
        for (size_t i = 0; i < count; ++i) {
            first_values[i] = first.data<float>()[i];
        }
    }
    stable_request.infer();
    const ov::Tensor second = stable_request.get_output_tensor();
    ASSERT_EQ(first.get_shape(), second.get_shape());
    for (size_t i = 0; i < first_values.size(); ++i) {
        EXPECT_EQ(first_values[i], second.data<float>()[i]) << "Non-deterministic output at " << i;
    }
}

// [S13] GenAI probe: the OpenVINO GenAI continuous-batching path forwards tensor parallelism as a
// *string* config entry, e.g. {"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"} (PLAN.md §3). This
// test emulates exactly that GenAI-style config to prove the property is parsed by the GPU plugin
// (string -> std::set<ModelDistributionPolicy> via ov::Any::as) and reaches the TP compile branch.
TEST_F(TensorParallelCompileGPU, GenAiStringConfigEnablesTensorParallel) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    // Same shape GenAI uses: a plain string value under the property's config key.
    ov::AnyMap genai_config = {{"MODEL_DISTRIBUTION_POLICY", "TENSOR_PARALLEL"}};

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, genai_config));

    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_EQ(ranks, (std::vector<int32_t>{0, 1})) << "GenAI string config did not enable tensor parallel";

    // The string is surfaced as the typed enum set (read-only report).
    auto policy = compiled_model.get_property(ov::hint::model_distribution_policy)
                      .as<std::set<ov::hint::ModelDistributionPolicy>>();
    EXPECT_EQ(policy.count(ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL), 1u);

    // The shard-detection pass must have run on this GenAI-style compile.
    auto seams = compiled_model.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    EXPECT_FALSE(seams.empty());
}

// [S13] GenAI-style stateful decode: compile through the GenAI string config, drive the model via
// the public *name-based* tensor API (set_tensor("input_ids")/get_tensor("logits")) like
// LLMPipeline does, run several dynamic-shape decode steps and compare against a single-GPU
// reference. Verifies the TP compile frame is compatible with how GenAI consumes a stateful model
// and that query_state()/set_state() round-trips the head-sharded KV state in that context.
TEST_F(TensorParallelCompileGPU, GenAiStyleStatefulRoundtripMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    const size_t heads = 4;
    const size_t head_dim = 2;
    auto ref_model = make_genai_style_stateful_model(heads, head_dim);
    auto tp_model = ref_model->clone();

    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    ov::AnyMap genai_config = {{"MODEL_DISTRIBUTION_POLICY", "TENSOR_PARALLEL"}};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, genai_config);

    // The named state variable must be head-sharded (aggregated at the GenAI boundary).
    auto state_seams = tp_compiled.get_property(ov::intel_gpu::tp_state_seams).as<std::vector<std::string>>();
    ASSERT_EQ(state_seams.size(), 1u);
    EXPECT_EQ(state_seams[0].rfind("HEAD_SHARDED:", 0), 0u) << "Unexpected state seam: " << state_seams[0];

    auto ref_request = ref_compiled.create_infer_request();
    auto tp_request = tp_compiled.create_infer_request();

    // A GenAI-style prefill+decode loop on a dynamic sequence axis.
    for (size_t t = 0; t < 4; ++t) {
        const size_t step = t + 1;
        ov::Tensor x(ov::element::f32, ov::Shape{1, 1, head_dim});
        auto* data = x.data<float>();
        for (size_t i = 0; i < head_dim; ++i) {
            data[i] = 0.2f * static_cast<float>(step) + static_cast<float>(i) - 0.4f;
        }
        // Name-based binding, exactly as GenAI/LLMPipeline drives a stateful decoder.
        ref_request.set_tensor("input_ids", x);
        tp_request.set_tensor("input_ids", x);
        ref_request.infer();
        tp_request.infer();
        const ov::Tensor ref_output = ref_request.get_tensor("logits");
        const ov::Tensor tp_output = tp_request.get_tensor("logits");
        ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
        ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
    }

    // GenAI round-trips the KV cache through query_state()/set_state() between steps.
    auto ref_states = ref_request.query_state();
    auto tp_states = tp_request.query_state();
    ASSERT_EQ(tp_states.size(), 1u);
    ASSERT_EQ(ref_states.size(), 1u);
    const ov::Tensor captured = tp_states[0].get_state();
    ASSERT_EQ(captured.get_shape(), (ov::Shape{1, heads, head_dim}));
    ov::test::utils::compare(ref_states[0].get_state(), captured, 1e-4f, 1e-4f);
    tp_states[0].set_state(captured);

    // One more step after the round-trip must still match the single-GPU reference.
    ov::Tensor x(ov::element::f32, ov::Shape{1, 1, head_dim});
    auto* data = x.data<float>();
    for (size_t i = 0; i < head_dim; ++i) {
        data[i] = -0.3f + static_cast<float>(i);
    }
    ref_request.set_tensor("input_ids", x);
    tp_request.set_tensor("input_ids", x);
    ref_request.infer();
    tp_request.infer();
    ov::test::utils::compare(ref_request.get_tensor("logits"), tp_request.get_tensor("logits"), 1e-4f, 1e-4f);
}

}  // namespace
