// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04] Experimental GPU tensor-parallel support.
// TPCoordinator enumerates the available GPUs and holds one per-rank execution context
// and configuration. It is the compile-time entry point used by Plugin::compile_model
// when a tensor-parallel distribution policy is requested.
// See PLAN.md §4.1, §4.2.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

namespace ov::intel_gpu {

/**
 * @brief Coordinates tensor-parallel compilation across several Intel GPUs.
 *
 * Unlike the regular GPU plugin flow (where one CompiledModel is bound to exactly one
 * device), the coordinator produces k independent per-rank contexts/configs, because
 * RemoteContextImpl only supports a single device (remote_context.cpp requires
 * devices.size() == 1). Each rank is later compiled by the existing single-device path.
 *
 * The coordinator itself does NOT build graphs or transport collectives - it only builds
 * the rank table and the (currently seam-less) TPPlan.
 */
class TPCoordinator {
public:
    /**
     * @brief One tensor-parallel rank: an independent device, context and configuration.
     */
    struct RankContext {
        int32_t rank = 0;
        std::string device_id;
        RemoteContextImpl::Ptr context;
        ExecutionConfig config;
    };

    /**
     * @brief Builds the coordinator from the enumerated rank contexts.
     *
     * @param ranks One entry per participating device. Must contain at least two ranks.
     * @throws ov::Exception if fewer than two ranks are provided.
     */
    explicit TPCoordinator(std::vector<RankContext> ranks);

    const std::vector<RankContext>& ranks() const { return m_ranks; }
    std::vector<RankContext>& ranks() { return m_ranks; }

    //! Builds a minimal compile-time plan: ranks + device ids, empty seam list.
    TPPlan make_plan() const;

    size_t num_ranks() const { return m_ranks.size(); }

private:
    std::vector<RankContext> m_ranks;
};

}  // namespace ov::intel_gpu
