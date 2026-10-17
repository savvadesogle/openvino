// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04] Experimental GPU tensor-parallel support.
// Implements TPCoordinator plus the tensor-parallel branch of Plugin::compile_model.
// No sharding or collective transport happens here yet (S06+ / S08+).
// See PLAN.md §4.1, §4.2.

#include "intel_gpu/plugin/tp/tp_coordinator.hpp"

#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_gpu {

TPCoordinator::TPCoordinator(std::vector<RankContext> ranks) : m_ranks(std::move(ranks)) {
    OPENVINO_ASSERT(m_ranks.size() >= 2,
                    "[GPU][TP] Tensor parallel requires at least 2 devices, but only ",
                    m_ranks.size(),
                    " available");
}

TPPlan TPCoordinator::make_plan() const {
    TPPlan plan;
    plan.ranks.reserve(m_ranks.size());
    plan.rank_device_ids.reserve(m_ranks.size());
    for (const auto& r : m_ranks) {
        plan.ranks.push_back(r.rank);
        plan.rank_device_ids.push_back(r.device_id);
    }
    // Seam classification is filled in by compile_model_tensor_parallel from the seams
    // detected by TPShardTransform (see get_tp_seams); make_plan only knows the rank table.
    return plan;
}

bool Plugin::is_tensor_parallel_requested(const ExecutionConfig& config) const {
    if (config.get_enable_tensor_parallel()) {
        return true;
    }
    const auto& policy = config.get_model_distribution_policy();
    return policy.count(ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL) > 0;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model_tensor_parallel(const std::shared_ptr<const ov::Model>& model,
                                                                          const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::compile_model::TensorParallel");

    // A tensor-parallel group spans all enumerated devices, so an explicit device id ("GPU.0") is ambiguous.
    OPENVINO_ASSERT(orig_config.find(ov::device::id.name()) == orig_config.end(),
                    "[GPU][TP] Tensor parallel cannot be requested for a specific device id (e.g. \"GPU.0\"). ",
                    "Use \"GPU\" to span all available devices");
    OPENVINO_ASSERT(m_device_map.size() >= 2,
                    "[GPU][TP] Tensor parallel requires at least 2 Intel GPUs, but only ",
                    m_device_map.size(),
                    " available");

    // 1) Enumerate the TP group and build one independent per-rank context/config per device.
    //    One RemoteContextImpl == one device (remote_context.cpp requires devices.size() == 1),
    //    so we deliberately create k contexts rather than a single multi-device one.
    std::vector<TPCoordinator::RankContext> rank_contexts;
    rank_contexts.reserve(m_device_map.size());
    int32_t rank = 0;
    for (const auto& device : m_device_map) {
        const std::string& device_id = device.first;
        TPCoordinator::RankContext rc;
        rc.rank = rank++;
        rc.device_id = device_id;
        rc.context = get_default_context(device_id);
        rc.config = m_configs_map.at(device_id);
        rc.config.set_user_property(orig_config, OptionVisibility::RELEASE);
        rank_contexts.push_back(std::move(rc));
    }

    TPCoordinator coordinator(std::move(rank_contexts));
    auto plan = coordinator.make_plan();

    // 2) Compile the model once per rank through the existing single-device path.
    //    In S06 TPShardTransform is an identity/analysis pass: every rank receives the full
    //    graph, but the detected seams are collected into the plan (see below).
    std::vector<std::shared_ptr<CompiledModel>> rank_models;
    rank_models.reserve(coordinator.num_ranks());
    bool seams_collected = false;
    for (auto& rc : coordinator.ranks()) {
        auto transformed_model = clone_and_transform_model(model, rc.config, rc.context);
        // [S06] TPShardTransform stamped the detected seams into the transformed model's
        // rt_info. All ranks transform the identical graph, so rank 0 is representative;
        // collect once to keep the plan deterministic and rank-independent.
        if (!seams_collected && rc.rank == 0) {
            plan.seams = get_tp_seams(transformed_model);
            seams_collected = true;
        }
        rc.config.finalize(rc.context.get(), transformed_model.get());
        rank_models.push_back(std::make_shared<CompiledModel>(transformed_model, shared_from_this(), rc.context, rc.config));
    }

    return std::make_shared<TPCompiledModel>(shared_from_this(), std::move(rank_models), std::move(plan));
}

}  // namespace ov::intel_gpu
