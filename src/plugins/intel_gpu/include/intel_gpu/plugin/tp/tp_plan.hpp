// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04/S06/S07/S08/S10/S16] Experimental GPU tensor-parallel support.
// Minimal tensor-parallel plan produced at compile time by the TPCoordinator.
// S06 adds seam detection: TPShardTransform classifies MatMul/FullyConnected nodes and the
// coordinator collects them into TPPlan.seams. S07 adds N-split weight slicing + all-gather
// output seams. S08 adds K-split weight/activation slicing + all-reduce(sum) output seams
// (fp32 accumulation; replicated bias/residual terms are added once by the reducer).
// S10 adds state-variable sharding (TPStateSeam): static, head-local state is head-sharded per
// rank and aggregated by TPInferRequest; everything else stays replicated.
// See PLAN.md §4.2, §4.3, §4.4, §7.4.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
class Model;
}  // namespace ov

namespace cldnn {
class BinaryOutputBuffer;
class BinaryInputBuffer;
}  // namespace cldnn

namespace ov::intel_gpu {

/**
 * @brief Kind of a tensor-parallel seam (boundary between two ranks that requires a
 * collective operation once sharding is actually applied).
 */
enum class TPSeamKind : uint8_t {
    NONE = 0,   //!< placeholder, unclassified node (no seam)
    N_SPLIT,    //!< weights split along output channels; ranks hold disjoint N slices.
                //!< Assembly = all-gather (concat, no arithmetic). PLAN §4.3.
    K_SPLIT,    //!< GEMM split along the contraction (K) dimension; partial results are
                //!< reduced with all-reduce(sum). PLAN §4.3. Bias is added once in the reducer.
    REPLICA,    //!< node replicated on every rank; no boundary and no collective.
};

/**
 * @brief Collective operation required at a seam once sharding is implemented (S07/S08).
 * Recorded by the analysis pass but NOT executed in S06.
 */
enum class TPCollectiveKind : uint8_t {
    NONE = 0,
    ALL_GATHER,     //!< N-split seam assembly (concatenation along the split axis)
    ALL_REDUCE_SUM, //!< K-split seam reduction (fp32 accumulation)
};

/**
 * @brief A single seam description.
 *
 * S06 populates node_name/node_type/kind/collective deterministically from the graph.
 * S07/S08 additionally fill split_axis/shard_dim_size when the node is really sliced.
 */
struct TPSeam {
    //! Friendly name of the detected node (stable for a given input model).
    std::string node_name;
    //! Node type string, e.g. "MatMul" / "FullyConnected" / "FullyConnectedCompressed".
    std::string node_type;
    //! Classified seam kind.
    TPSeamKind kind = TPSeamKind::NONE;
    //! Collective required at this seam (S07/S08; informational in S06).
    TPCollectiveKind collective = TPCollectiveKind::NONE;
    //! Axis to split/reduce along. Placeholder (-1 = undecided), filled in S07/S08.
    int64_t split_axis = -1;
    //! Size of the dimension being split. Placeholder (-1 = unknown), filled in S07/S08.
    int64_t shard_dim_size = -1;
};

/**
 * @brief Kind of a replicated additive term that each rank adds to a K-split partial GEMM
 * output (a bias / residual that must be counted exactly ONCE by the reducer, see PLAN §4.3).
 */
enum class TPReplicatedAddKind : uint8_t {
    NONE = 0,    //!< no replicated additive term
    CONSTANT,    //!< a constant already broadcast-expanded to the model output shape
    INPUT,       //!< a model input tensor (e.g. a residual stream) replicated on every rank
};

/**
 * @brief A replicated additive term of a K-split (all-reduce sum) output seam.
 *
 * Each rank computes `partial_r + term` and the reducer computes
 * `full = sum_r partial_r + term` as `sum_r (partial_r + term) - (num_ranks - 1) * term`.
 * Constants are pre-expanded to the (static) model output shape at compile time; inputs are
 * broadcast from their runtime shape by aligning trailing dimensions (S08).
 */
struct TPReplicatedAdd {
    TPReplicatedAddKind kind = TPReplicatedAddKind::NONE;
    //! [CONSTANT] broadcast-expanded fp32 values, row-major, length == product(output shape).
    std::vector<float> values;
    //! [INPUT] index of the replicated model input (into ov::Model::inputs()/get_parameters()).
    size_t input_index = 0;
};

/**
 * @brief Describes a model output that is produced (possibly through a chain of purely
 * element-wise operations) by a sharded GEMM and therefore must be assembled on rank 0.
 *
 *  - ALL_GATHER (N-split, S07): each rank holds a disjoint slice of the last axis; the full
 *    tensor is assembled by concatenation. With an equal split the rank-r slice spans
 *    [rank_r * (shard_dim_size / num_ranks), (rank_r + 1) * (shard_dim_size / num_ranks))
 *    along split_axis.
 *  - ALL_REDUCE_SUM (K-split, S08): each rank holds a partial GEMM sum of the full output
 *    shape; the full tensor is the fp32 sum of the rank partials (with the replicated terms
 *    subtracted once).
 */
struct TPOutputSeam {
    //! Index of the model output (into ov::Model::outputs()).
    size_t output_index = 0;
    //! Collective required to assemble this output (ALL_GATHER for N-split, ALL_REDUCE_SUM for K-split).
    TPCollectiveKind collective = TPCollectiveKind::ALL_GATHER;
    //! Axis along which the full tensor is sharded (negative = counted from the end, -1 = last).
    int64_t split_axis = -1;
    //! Size of the sharded dimension in the FULL (unsharded) tensor.
    int64_t shard_dim_size = -1;
    //! Friendly name of the sharded GEMM that produces this output (debug/introspection).
    std::string producer_node;
    //! [S08, ALL_REDUCE_SUM] replicated additive terms added once by the reducer (bias/residual).
    std::vector<TPReplicatedAdd> replicated_adds;
};

/**
 * @brief How a stateful variable (attention KV cache / recurrent state) is distributed across
 * tensor-parallel ranks (S10). PLAN §4.3.
 */
enum class TPStateShardKind : uint8_t {
    REPLICATED = 0,  //!< identical state on every rank; aggregation = rank-0 copy
    HEAD_SHARDED,    //!< state split along a head axis; aggregation = concatenation along that axis
};

/**
 * @brief Describes how a single model state variable (ReadValue/Assign) is distributed.
 *
 * For HEAD_SHARDED variables each rank keeps a contiguous `[axis] = full_dim / num_ranks` slice
 * of the (static) state; TPInferRequest::query_state() concatenates the per-rank slices back into
 * the full state along @p axis, and set_state()/reset() distribute accordingly. REPLICATED state
 * is left untouched (rank 0 authoritative). See docs/notes/state-inventory.md.
 */
struct TPStateSeam {
    //! ov::op::util::VariableInfo::variable_id (also the runtime IVariableState name).
    std::string variable_id;
    //! Sharding kind.
    TPStateShardKind kind = TPStateShardKind::REPLICATED;
    //! Head axis of the FULL state shape (non-negative index); -1 for REPLICATED.
    int64_t axis = -1;
    //! Size of the head axis in the FULL (unsharded) state; -1 for REPLICATED. Always static
    //! (S12 head-shards only along a static, rank-divisible axis even when other dims are dynamic).
    int64_t full_dim = -1;
    //! Full (unsharded) state shape. May carry dynamic dimensions (batch/sequence) as in the real
    //! Qwen3.5 states (`?,4,?,256`, `?,32,128,128`); only the head axis is guaranteed static (S12).
    ov::PartialShape full_shape;
    //! Human-readable reason (debug/introspection): why replicated or head-sharded.
    std::string reason;
};

/**
 * @brief Compile-time tensor-parallel plan.
 *
 * Describes the participating ranks (and their device ids), the classified seams detected by
 * TPShardTransform and the model outputs that require a collective to be assembled on rank 0
 * (all-gather for N-split, all-reduce sum for K-split, S07/S08) plus the state-variable sharding
 * decisions (S10). The structure is deliberately minimal so that later stages can extend it
 * without touching the public API.
 */
struct TPPlan {
    //! Rank indices, e.g. {0, 1} for a 2-GPU TP group.
    std::vector<int32_t> ranks;
    //! Device id (as in ov::available_devices) materializing each rank; ranks[i] lives on
    //! rank_device_ids[i].
    std::vector<std::string> rank_device_ids;
    //! Classified seams (one entry per detected MatMul/FC), in deterministic graph order.
    std::vector<TPSeam> seams;
    //! Model outputs that must be assembled from per-rank slices (all-gather, S07).
    std::vector<TPOutputSeam> output_seams;
    //! State variables and their sharding (S10), in deterministic variable-id order.
    std::vector<TPStateSeam> state_seams;

    bool tp_enabled() const { return ranks.size() > 1; }
    size_t num_ranks() const { return ranks.size(); }
    //! Number of seams that will actually require a collective (N_SPLIT/K_SPLIT).
    size_t num_sharded_seams() const;
    //! Finds the assembly descriptor for a model output index (nullptr if it is replicated).
    const TPOutputSeam* find_output_seam(size_t output_index) const;
    //! Finds the sharding descriptor for a state variable id (nullptr if unknown/replicated).
    const TPStateSeam* find_state_seam(const std::string& variable_id) const;
};

//! Stable string encodings used for logging and for the ov::intel_gpu::tp_seams property.
const char* to_string(TPSeamKind kind);
const char* to_string(TPCollectiveKind collective);
//! Encodes a seam as "<kind>:<collective>:<node_type>:<node_name>".
std::string tp_seam_to_string(const TPSeam& seam);

//! Model rt_info key under which TPShardTransform stores the detected seams.
//! The value has the plain type std::vector<TPSeam> (no registered attribute needed).
const char* tp_seams_rt_info_key();

//! Model rt_info key under which TPShardTransform stores the outputs that require an
//! all-gather once the N-split weights have been sliced (S07).
const char* tp_output_seams_rt_info_key();

//! Encodes an output seam as "<producer>:<split_axis>:<shard_dim_size>" (logging/introspection).
std::string tp_output_seam_to_string(const TPOutputSeam& seam);

//! Encodes a state seam as "<kind>:<axis>:<full_dim>:<variable_id>" (logging/introspection).
std::string tp_state_seam_to_string(const TPStateSeam& seam);

//! Stores the detected seams in the model rt_info (used by TPShardTransform).
void set_tp_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPSeam> seams);
//! Reads the seams previously stored by TPShardTransform (empty if none / non-TP model).
std::vector<TPSeam> get_tp_seams(const std::shared_ptr<const ov::Model>& model);

//! Stores the detected all-gather outputs in the model rt_info (used by TPShardTransform).
void set_tp_output_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPOutputSeam> seams);
//! Reads the all-gather outputs previously stored by TPShardTransform (empty if none).
std::vector<TPOutputSeam> get_tp_output_seams(const std::shared_ptr<const ov::Model>& model);

//! [S10] Stores the detected state-variable sharding in the model rt_info (used by TPShardTransform).
void set_tp_state_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPStateSeam> seams);
//! [S10] Reads the state-variable sharding previously stored by TPShardTransform (empty if none).
std::vector<TPStateSeam> get_tp_state_seams(const std::shared_ptr<const ov::Model>& model);

//! [S16] Magic/version of the per-rank tensor-parallel compiled-model cache envelope.
//! The envelope wraps one independently exported single-device blob per rank plus the TPPlan, so
//! the ranks of a TP group never share a blob (see TPCompiledModel::export_model).
constexpr uint32_t TENSOR_PARALLEL_BLOB_MAGIC = 0x54505342u;  //!< "TPSB"
constexpr uint32_t TENSOR_PARALLEL_BLOB_VERSION = 1u;

//! [S16] Serializes the tensor-parallel plan into the compiled-model cache envelope.
void save_tp_plan(cldnn::BinaryOutputBuffer& ob, const TPPlan& plan);
//! [S16] Deserializes a tensor-parallel plan previously written by save_tp_plan.
void load_tp_plan(cldnn::BinaryInputBuffer& ib, TPPlan& plan);

}  // namespace ov::intel_gpu
