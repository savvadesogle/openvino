// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04/S06] Experimental GPU tensor-parallel support.
// Minimal tensor-parallel plan produced at compile time by the TPCoordinator.
// S06 adds seam detection: TPShardTransform (identity/analysis pass) classifies
// MatMul/FullyConnected nodes and the coordinator collects them into TPPlan.seams.
// Seams are metadata only at this stage - NO weight slicing and NO collectives are
// performed yet (S07/S08). See PLAN.md §4.2, §4.3, §4.4.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

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
 * The remaining fields are placeholders reserved for S07/S08 (real slicing) and are NOT
 * consumed by any runtime behavior yet.
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
 * @brief Compile-time tensor-parallel plan.
 *
 * Describes the participating ranks (and their device ids) plus the classified seams
 * detected by TPShardTransform. The structure is deliberately minimal so that later
 * stages can extend it without touching the public API.
 */
struct TPPlan {
    //! Rank indices, e.g. {0, 1} for a 2-GPU TP group.
    std::vector<int32_t> ranks;
    //! Device id (as in ov::available_devices) materializing each rank; ranks[i] lives on
    //! rank_device_ids[i].
    std::vector<std::string> rank_device_ids;
    //! Classified seams (one entry per detected MatMul/FC), in deterministic graph order.
    std::vector<TPSeam> seams;

    bool tp_enabled() const { return ranks.size() > 1; }
    size_t num_ranks() const { return ranks.size(); }
    //! Number of seams that will actually require a collective (N_SPLIT/K_SPLIT).
    size_t num_sharded_seams() const;
};

//! Stable string encodings used for logging and for the ov::intel_gpu::tp_seams property.
const char* to_string(TPSeamKind kind);
const char* to_string(TPCollectiveKind collective);
//! Encodes a seam as "<kind>:<collective>:<node_type>:<node_name>".
std::string tp_seam_to_string(const TPSeam& seam);

//! Model rt_info key under which TPShardTransform stores the detected seams.
//! The value has the plain type std::vector<TPSeam> (no registered attribute needed).
const char* tp_seams_rt_info_key();

//! Stores the detected seams in the model rt_info (used by TPShardTransform).
void set_tp_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPSeam> seams);
//! Reads the seams previously stored by TPShardTransform (empty if none / non-TP model).
std::vector<TPSeam> get_tp_seams(const std::shared_ptr<const ov::Model>& model);

}  // namespace ov::intel_gpu
