// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04] GATE test for the GPU tensor-parallel compile path (Stage 1.1).
// Milestone S04: Plugin::compile_model with model_distribution_policy{TENSOR_PARALLEL} over
// a 2-GPU mini network (Parameter -> MatMul -> Result) must succeed and report k=2 ranks via
// ov::intel_gpu::tp_ranks. Execution parity is NOT checked here (S05).
// See PLAN.md §4.1, §4.2, §5 (S04).

#include <set>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

// A trivial single-FullyConnected-like model: Parameter -> MatMul -> Result.
std::shared_ptr<ov::Model> make_fc_mini_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, {1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_fc_mini");
}

// Number of individually addressable GPU devices (excludes sub-devices such as "GPU.0.0").
size_t count_gpu_devices(ov::Core& core) {
    size_t count = 0;
    for (const auto& device : core.get_available_devices()) {
        // "GPU.N" - exactly one dot, no further sub-device suffix.
        if (device.rfind("GPU.", 0) == 0 && device.find('.', 4) == std::string::npos) {
            ++count;
        }
    }
    return count;
}

class TensorParallelCompileGPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = make_fc_mini_model();
    }
};

TEST_F(TensorParallelCompileGPU, CompilesMiniModelOnTwoGpus) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props));
    ASSERT_NE(compiled_model, nullptr);

    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    ASSERT_EQ(ranks.size(), 2u) << "Expected exactly 2 tensor-parallel ranks";
    EXPECT_EQ(ranks, (std::vector<int32_t>{0, 1}));
}

TEST_F(TensorParallelCompileGPU, InternalFlagAlsoEnablesTensorParallel) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    ov::AnyMap props = {ov::intel_gpu::enable_tensor_parallel(true)};

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props));

    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_EQ(ranks.size(), 2u);
}

TEST_F(TensorParallelCompileGPU, DefaultCompileHasNoRanks) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU));
    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_TRUE(ranks.empty());
}

TEST_F(TensorParallelCompileGPU, RejectsExplicitDeviceId) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    // Tensor parallelism spans all devices, so "GPU.0" is ambiguous and must be rejected.
    EXPECT_THROW(core.compile_model(shared_model, "GPU.0", props), ov::Exception);
}

TEST_F(TensorParallelCompileGPU, RejectsPipelineParallel) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> pp_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(pp_policy)};

    EXPECT_THROW(core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props), ov::Exception);
}

}  // namespace
