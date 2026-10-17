// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04] Experimental GPU tensor-parallel support.
// TPCompiledModel wraps k per-rank CompiledModel instances (one per device) and a TPPlan.
// In S04 it is a skeleton: it only reports the tensor-parallel rank table (ov::intel_gpu::tp_ranks)
// and delegates inference to the rank-0 compiled model. Correct k-rank execution
// (input replication, rank-0 outputs, device-side seams) is implemented in S05+.
// See PLAN.md §4.4.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace ov::intel_gpu {

/**
 * @brief Compiled model spanning several GPUs with a tensor-parallel distribution.
 *
 * Owns k independent per-rank CompiledModel objects (each bound to its own device/engine).
 * S04 scope: compile-time skeleton + tp_ranks report. Inference currently delegates to rank 0.
 */
class TPCompiledModel : public ov::ICompiledModel {
public:
    using Ptr = std::shared_ptr<TPCompiledModel>;

    TPCompiledModel(const std::shared_ptr<const ov::IPlugin>& plugin,
                    std::vector<std::shared_ptr<CompiledModel>> ranks,
                    TPPlan plan);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    ov::Any get_property(const std::string& name) const override;
    void set_property(const ov::AnyMap& properties) override;

    const std::vector<ov::Output<const ov::Node>>& outputs() const override;
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

    const TPPlan& get_tp_plan() const { return m_plan; }
    const std::vector<std::shared_ptr<CompiledModel>>& get_rank_models() const { return m_ranks; }

private:
    std::vector<std::shared_ptr<CompiledModel>> m_ranks;
    TPPlan m_plan;
};

}  // namespace ov::intel_gpu
