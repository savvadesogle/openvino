// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04] Experimental GPU tensor-parallel support.
// Minimal tensor-parallel plan produced at compile time by the TPCoordinator.
// Sharding (TPShardTransform) and collective transport are NOT part of this stage:
// the seam classification below is an intentionally empty skeleton to be populated
// in later stages (S06+). See PLAN.md §4.2, §4.4.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ov::intel_gpu {

/**
 * @brief Kind of a tensor-parallel seam (boundary between two ranks that requires a
 * collective operation). Empty in S04 - sharding and seam detection start in S06.
 */
enum class TPSeamKind : uint8_t {
    NONE = 0,   //!< placeholder, no seam
};

/**
 * @brief A single seam description. Skeleton only; the fields become meaningful once
 * TPShardTransform starts marking the graph (S06+).
 */
struct TPSeam {
    std::string node_name;
    TPSeamKind kind = TPSeamKind::NONE;
};

/**
 * @brief Compile-time tensor-parallel plan.
 *
 * In S04 it only describes the participating ranks (and their device ids); the seam
 * list stays empty because no sharding is applied yet. The structure is deliberately
 * minimal so that later stages can extend it without touching the public API.
 */
struct TPPlan {
    //! Rank indices, e.g. {0, 1} for a 2-GPU TP group.
    std::vector<int32_t> ranks;
    //! Device id (as in ov::available_devices) materializing each rank; ranks[i] lives on
    //! rank_device_ids[i].
    std::vector<std::string> rank_device_ids;
    //! Classified seams between ranks; empty until sharding is implemented (S06+).
    std::vector<TPSeam> seams;

    bool tp_enabled() const { return ranks.size() > 1; }
    size_t num_ranks() const { return ranks.size(); }
};

}  // namespace ov::intel_gpu
