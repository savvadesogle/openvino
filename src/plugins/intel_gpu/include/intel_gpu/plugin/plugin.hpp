// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iplugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include <map>
#include <string>
#include <memory>

namespace ov::intel_gpu {

class TPCoordinator;

class Plugin : public ov::IPlugin {
private:
    std::string m_default_device_id = "0";
    std::map<std::string, cldnn::device::ptr> m_device_map;
    std::map<std::string, ExecutionConfig> m_configs_map;
    ov::AnyMap m_compiled_model_runtime_properties;

    mutable std::map<std::string, std::shared_ptr<RemoteContextImpl>> m_default_contexts;
    mutable std::once_flag m_default_contexts_once;

    std::map<std::string, std::shared_ptr<RemoteContextImpl>> get_default_contexts() const;

    // Rename our enumerated devices to the ids Core assigned (ov::internal::device_id_map).
    void apply_device_id_map(const std::map<std::string, std::string>& id_map);

    std::shared_ptr<ov::Model> clone_and_transform_model(const std::shared_ptr<const ov::Model>& network,
                                                         ExecutionConfig& config,
                                                         const std::shared_ptr<RemoteContextImpl>& context) const;
    void transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config, const std::shared_ptr<RemoteContextImpl>& context) const;
    void register_primitives() const;
    bool is_weightless_cache_attributes_set(const std::shared_ptr<const ov::Model>& model) const;
    void set_weightless_cache_attributes(const std::shared_ptr<const ov::Model>& model) const;
    void create_weightless_cache_attributes(const std::shared_ptr<const ov::Model>& model, ExecutionConfig& config) const;
    std::string get_device_id_from_config(const ov::AnyMap& config) const;
    std::string get_device_id(const ov::AnyMap& config) const;
    std::shared_ptr<RemoteContextImpl> get_default_context(const std::string& device_id, bool initialize = true) const;

    // [TP-GPU, S04] Tensor-parallel compile path: returns true when the config asks for TP
    // (either via ov::hint::model_distribution_policy{TENSOR_PARALLEL} or the internal
    // ov::intel_gpu::enable_tensor_parallel flag).
    bool is_tensor_parallel_requested(const ExecutionConfig& config) const;
    // Builds a tensor-parallel group over all enumerated GPUs: k per-rank contexts/configs,
    // one (currently unsharded) compiled graph per rank, wrapped in a TPCompiledModel.
    std::shared_ptr<ov::ICompiledModel> compile_model_tensor_parallel(const std::shared_ptr<const ov::Model>& model,
                                                                      const ov::AnyMap& orig_config) const;

    // [TP-GPU, S16] Imports the per-rank envelope produced by TPCompiledModel::export_model: one
    // independent single-device blob per rank, each reconstructed on its own device context. Keeps
    // the ranks of a TP group from sharing a compiled blob.
    std::shared_ptr<ov::ICompiledModel> import_model_tensor_parallel(std::istream& model,
                                                                     const ov::AnyMap& orig_config) const;
    // [TP-GPU, S16] Imports a single-device blob (the ordinary GPU cache path). Factored out so the
    // tensor-parallel import can reuse it per rank; behaviour for the non-TP path is unchanged.
    std::shared_ptr<ov::ICompiledModel> import_single_device_model(std::istream& model,
                                                                   const std::shared_ptr<RemoteContextImpl>& context_impl,
                                                                   ExecutionConfig& config,
                                                                   bool loaded_from_cache,
                                                                   const size_t* tensor_base) const;

    std::vector<ov::PropertyName> get_caching_properties() const;
    std::vector<ov::PropertyName> get_supported_properties() const;
    std::vector<ov::PropertyName> get_supported_internal_properties() const;
    std::vector<std::string> get_device_capabilities(const cldnn::device_info& info) const;
    uint32_t get_optimal_batch_size(const ov::AnyMap& options) const;
    uint32_t get_max_batch_size(const ov::AnyMap& options) const;

    bool is_metric(const std::string& name) const;
    ov::Any get_metric(const std::string& name, const ov::AnyMap& arguments) const;

public:
    Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;
};

}  // namespace ov::intel_gpu
