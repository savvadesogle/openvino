// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S06] Experimental GPU tensor-parallel support.
// TPPlan helpers: string encodings and rt_info transport of detected seams.
// S10 adds state-variable sharding transport (TPStateSeam).
// S16 adds binary (de)serialization of the whole TPPlan for the per-rank cache envelope.
// See PLAN.md §4.2, §4.3.

#include "intel_gpu/plugin/tp/tp_plan.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
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

// [S16] The compiled-model cache envelope for a tensor-parallel group (see
// TPCompiledModel::export_model / Plugin::import_model_tensor_parallel). The plan is serialized
// with the same cldnn binary buffers used by the rest of the GPU cache blob so no new format is
// invented for the primitives/constants: enums are written as their underlying uint8_t, shapes via
// the PartialShape serializer, and replicated additive values as a plain fp32 vector.
namespace {
void save_seam(cldnn::BinaryOutputBuffer& ob, const TPSeam& seam) {
    ob << seam.node_name;
    ob << seam.node_type;
    ob << static_cast<uint8_t>(seam.kind);
    ob << static_cast<uint8_t>(seam.collective);
    ob << seam.split_axis;
    ob << seam.shard_dim_size;
}

void load_seam(cldnn::BinaryInputBuffer& ib, TPSeam& seam) {
    ib >> seam.node_name;
    ib >> seam.node_type;
    uint8_t kind = 0;
    uint8_t collective = 0;
    ib >> kind;
    ib >> collective;
    seam.kind = static_cast<TPSeamKind>(kind);
    seam.collective = static_cast<TPCollectiveKind>(collective);
    ib >> seam.split_axis;
    ib >> seam.shard_dim_size;
}

void save_output_seam(cldnn::BinaryOutputBuffer& ob, const TPOutputSeam& seam) {
    ob << seam.output_index;
    ob << static_cast<uint8_t>(seam.collective);
    ob << seam.split_axis;
    ob << seam.shard_dim_size;
    ob << seam.producer_node;
    ob << seam.replicated_adds.size();
    for (const auto& add : seam.replicated_adds) {
        ob << static_cast<uint8_t>(add.kind);
        ob << add.values;
        ob << add.input_index;
    }
}

void load_output_seam(cldnn::BinaryInputBuffer& ib, TPOutputSeam& seam) {
    ib >> seam.output_index;
    uint8_t collective = 0;
    ib >> collective;
    seam.collective = static_cast<TPCollectiveKind>(collective);
    ib >> seam.split_axis;
    ib >> seam.shard_dim_size;
    ib >> seam.producer_node;
    size_t num_adds = 0;
    ib >> num_adds;
    seam.replicated_adds.resize(num_adds);
    for (auto& add : seam.replicated_adds) {
        uint8_t kind = 0;
        ib >> kind;
        add.kind = static_cast<TPReplicatedAddKind>(kind);
        ib >> add.values;
        ib >> add.input_index;
    }
}

void save_state_seam(cldnn::BinaryOutputBuffer& ob, const TPStateSeam& seam) {
    ob << seam.variable_id;
    ob << static_cast<uint8_t>(seam.kind);
    ob << seam.axis;
    ob << seam.full_dim;
    ob << seam.full_shape;
    ob << seam.reason;
}

void load_state_seam(cldnn::BinaryInputBuffer& ib, TPStateSeam& seam) {
    ib >> seam.variable_id;
    uint8_t kind = 0;
    ib >> kind;
    seam.kind = static_cast<TPStateShardKind>(kind);
    ib >> seam.axis;
    ib >> seam.full_dim;
    ib >> seam.full_shape;
    ib >> seam.reason;
}
}  // namespace

void save_tp_plan(cldnn::BinaryOutputBuffer& ob, const TPPlan& plan) {
    ob << plan.ranks;
    ob << plan.rank_device_ids;
    ob << plan.seams.size();
    for (const auto& seam : plan.seams) {
        save_seam(ob, seam);
    }
    ob << plan.output_seams.size();
    for (const auto& seam : plan.output_seams) {
        save_output_seam(ob, seam);
    }
    ob << plan.state_seams.size();
    for (const auto& seam : plan.state_seams) {
        save_state_seam(ob, seam);
    }
}

void load_tp_plan(cldnn::BinaryInputBuffer& ib, TPPlan& plan) {
    ib >> plan.ranks;
    ib >> plan.rank_device_ids;
    size_t num_seams = 0;
    ib >> num_seams;
    plan.seams.resize(num_seams);
    for (auto& seam : plan.seams) {
        load_seam(ib, seam);
    }
    size_t num_output_seams = 0;
    ib >> num_output_seams;
    plan.output_seams.resize(num_output_seams);
    for (auto& seam : plan.output_seams) {
        load_output_seam(ib, seam);
    }
    size_t num_state_seams = 0;
    ib >> num_state_seams;
    plan.state_seams.resize(num_state_seams);
    for (auto& seam : plan.state_seams) {
        load_state_seam(ib, seam);
    }
}

}  // namespace ov::intel_gpu
