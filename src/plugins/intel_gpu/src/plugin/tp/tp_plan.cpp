// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06] Experimental GPU tensor-parallel support.
// TPPlan helpers: string encodings and rt_info transport of detected seams.
// S10 adds state-variable sharding transport (TPStateSeam).
// See PLAN.md §4.2, §4.3.

#include "intel_gpu/plugin/tp/tp_plan.hpp"

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu {

namespace {
constexpr const char* TP_SEAMS_RT_INFO_KEY = "intel_gpu_tp_seams";
constexpr const char* TP_OUTPUT_SEAMS_RT_INFO_KEY = "intel_gpu_tp_output_seams";
constexpr const char* TP_STATE_SEAMS_RT_INFO_KEY = "intel_gpu_tp_state_seams";

const char* state_shard_kind_to_string(TPStateShardKind kind) {
    switch (kind) {
        case TPStateShardKind::REPLICATED: return "REPLICATED";
        case TPStateShardKind::HEAD_SHARDED: return "HEAD_SHARDED";
    }
    return "UNKNOWN";
}
}  // namespace

size_t TPPlan::num_sharded_seams() const {
    size_t count = 0;
    for (const auto& seam : seams) {
        if (seam.kind == TPSeamKind::N_SPLIT || seam.kind == TPSeamKind::K_SPLIT) {
            ++count;
        }
    }
    return count;
}

const TPOutputSeam* TPPlan::find_output_seam(size_t output_index) const {
    for (const auto& seam : output_seams) {
        if (seam.output_index == output_index) {
            return &seam;
        }
    }
    return nullptr;
}

const TPStateSeam* TPPlan::find_state_seam(const std::string& variable_id) const {
    for (const auto& seam : state_seams) {
        if (seam.variable_id == variable_id && seam.kind != TPStateShardKind::REPLICATED) {
            return &seam;
        }
    }
    return nullptr;
}

const char* to_string(TPSeamKind kind) {
    switch (kind) {
        case TPSeamKind::NONE: return "NONE";
        case TPSeamKind::N_SPLIT: return "N_SPLIT";
        case TPSeamKind::K_SPLIT: return "K_SPLIT";
        case TPSeamKind::REPLICA: return "REPLICA";
    }
    return "UNKNOWN";
}

const char* to_string(TPCollectiveKind collective) {
    switch (collective) {
        case TPCollectiveKind::NONE: return "NONE";
        case TPCollectiveKind::ALL_GATHER: return "ALL_GATHER";
        case TPCollectiveKind::ALL_REDUCE_SUM: return "ALL_REDUCE_SUM";
    }
    return "UNKNOWN";
}

std::string tp_seam_to_string(const TPSeam& seam) {
    return std::string(to_string(seam.kind)) + ":" + to_string(seam.collective) + ":" +
           seam.node_type + ":" + seam.node_name;
}

std::string tp_output_seam_to_string(const TPOutputSeam& seam) {
    return std::string(to_string(seam.collective)) + ":" + seam.producer_node + ":" +
           std::to_string(seam.split_axis) + ":" + std::to_string(seam.shard_dim_size) + ":" +
           std::to_string(seam.replicated_adds.size());
}

std::string tp_state_seam_to_string(const TPStateSeam& seam) {
    return std::string(state_shard_kind_to_string(seam.kind)) + ":" + std::to_string(seam.axis) + ":" +
           std::to_string(seam.full_dim) + ":" + seam.variable_id;
}

const char* tp_seams_rt_info_key() {
    return TP_SEAMS_RT_INFO_KEY;
}

const char* tp_output_seams_rt_info_key() {
    return TP_OUTPUT_SEAMS_RT_INFO_KEY;
}

void set_tp_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPSeam> seams) {
    if (!model) {
        return;
    }
    model->get_rt_info()[TP_SEAMS_RT_INFO_KEY] = std::move(seams);
}

std::vector<TPSeam> get_tp_seams(const std::shared_ptr<const ov::Model>& model) {
    if (!model) {
        return {};
    }
    const auto& rt_info = model->get_rt_info();
    auto it = rt_info.find(TP_SEAMS_RT_INFO_KEY);
    if (it == rt_info.end()) {
        return {};
    }
    return it->second.as<std::vector<TPSeam>>();
}

void set_tp_output_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPOutputSeam> seams) {
    if (!model) {
        return;
    }
    model->get_rt_info()[TP_OUTPUT_SEAMS_RT_INFO_KEY] = std::move(seams);
}

std::vector<TPOutputSeam> get_tp_output_seams(const std::shared_ptr<const ov::Model>& model) {
    if (!model) {
        return {};
    }
    const auto& rt_info = model->get_rt_info();
    auto it = rt_info.find(TP_OUTPUT_SEAMS_RT_INFO_KEY);
    if (it == rt_info.end()) {
        return {};
    }
    return it->second.as<std::vector<TPOutputSeam>>();
}

void set_tp_state_seams(const std::shared_ptr<ov::Model>& model, std::vector<TPStateSeam> seams) {
    if (!model) {
        return;
    }
    model->get_rt_info()[TP_STATE_SEAMS_RT_INFO_KEY] = std::move(seams);
}

std::vector<TPStateSeam> get_tp_state_seams(const std::shared_ptr<const ov::Model>& model) {
    if (!model) {
        return {};
    }
    const auto& rt_info = model->get_rt_info();
    auto it = rt_info.find(TP_STATE_SEAMS_RT_INFO_KEY);
    if (it == rt_info.end()) {
        return {};
    }
    return it->second.as<std::vector<TPStateSeam>>();
}

}  // namespace ov::intel_gpu
