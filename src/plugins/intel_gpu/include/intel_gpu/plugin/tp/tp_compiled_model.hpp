// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05/S14] Experimental GPU tensor-parallel support.
// TPCompiledModel wraps k per-rank CompiledModel instances (one per device) and a TPPlan.
// It reports the tensor-parallel rank table (ov::intel_gpu::tp_ranks) and creates a
// TPInferRequest that executes the replicated graph on all k ranks (rank-0 outputs).
// S14 adds a dedicated k-stream rank scheduler executor so per-rank sub-requests run in
// parallel (one stream per device) instead of serially.
// Sharding and device-side seams are implemented in S06+.
// See PLAN.md §4.4.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace ov {
namespace threading {
class IStreamsExecutor;
}  // namespace threading
}  // namespace ov

namespace ov::intel_gpu {

class TPInferRequest;

/**
 * @brief Compiled model spanning several GPUs with a tensor-parallel distribution.
 *
 * Owns k independent per-rank CompiledModel objects (each bound to its own device/engine).
 * S05 scope: create_infer_request()/create_sync_infer_request() return TPInferRequest,
 * which runs the replicated graph on every rank and exposes rank-0 outputs.
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

    // [S14] Shared rank scheduler: a k-stream executor (one stream per TP rank) used by
    // TPInferRequest to run the per-rank sub-requests concurrently. Owned for the lifetime of
    // the compiled model so no threads are created per inference (decode calls infer() per token).
    const std::shared_ptr<ov::threading::IStreamsExecutor>& get_rank_executor() const { return m_rank_executor; }

private:
    // Builds m_outputs: full-shape ports for N-split assembly outputs, native ports otherwise.
    void build_output_ports();

    std::vector<std::shared_ptr<CompiledModel>> m_ranks;
    TPPlan m_plan;

    // [S14] One stream per rank; the k rank sub-requests are dispatched here via run_and_wait.
    std::shared_ptr<ov::threading::IStreamsExecutor> m_rank_executor;

    // [S07] Model outputs as exposed to the user. Outputs produced by an N-split GEMM are
    // advertised at their FULL (unsharded) shape even though each rank only materializes its
    // slice; TPInferRequest assembles the full tensor from the per-rank slices on rank 0.
    // Keeps alive the lightweight synthetic nodes backing the reshaped output ports.
    std::vector<ov::Output<const ov::Node>> m_outputs;
    std::vector<std::shared_ptr<ov::Node>> m_synthetic_output_nodes;
};

}  // namespace ov::intel_gpu
