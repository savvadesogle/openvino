// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S04/S05] GATE tests for the GPU tensor-parallel compile + infer path.
// S04: Plugin::compile_model with model_distribution_policy{TENSOR_PARALLEL} over a 2-GPU
//      mini network (Parameter -> MatMul -> Result) must succeed and report k=2 ranks via
//      ov::intel_gpu::tp_ranks.
// S05: TPInferRequest executes the replicated graph on both ranks and exposes rank-0 outputs;
//      the result must match a single-GPU reference inference (e2e parity).
// S06: TPShardTransform (identity/analysis) detects MatMul/FullyConnected seams and reports
//      them via ov::intel_gpu::tp_seams; the graph/numbers are unchanged (S05 parity holds).
// See PLAN.md §4.1, §4.2, §4.3, §4.4, §5 (S04, S05, S06).

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
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

// A projection whose (partial) output is added back to a residual stream: the heuristic in
// TPShardTransform must classify it as K_SPLIT (all-reduce(sum)) rather than N_SPLIT.
std::shared_ptr<ov::Model> make_residual_projection_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, {1});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto add = std::make_shared<ov::op::v1::Add>(matmul, matmul);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_residual_proj");
}

// Same topology but with distinctive, non-uniform weights: parity mismatches caused by a rank
// not receiving the input (or the wrong rank's output being read) turn into large numeric deltas.
std::shared_ptr<ov::Model> make_tp_parity_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    std::vector<float> weights_data(8 * 16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        const float mag = 0.05f * static_cast<float>((i % 7) + 1);
        weights_data[i] = (i % 2 == 0) ? mag : -mag;
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 16}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "tp_parity");
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

// [S05] GATE: tensor-parallel inference (2 GPUs) must match single-GPU inference on the same
// replicated mini model. Ranks are identical at this stage, so no sharding arithmetic is involved.
TEST_F(TensorParallelCompileGPU, TensorParallelInferMatchesSingleGpu) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto ref_model = make_tp_parity_model();
    auto tp_model = ref_model->clone();

    ov::Tensor input(ov::element::f32, ov::Shape{1, 8});
    {
        auto* data = input.data<float>();
        for (size_t i = 0; i < 8; ++i) {
            data[i] = static_cast<float>(i) - 3.5f;
        }
    }

    // Reference: plain single-GPU inference.
    auto ref_compiled = core.compile_model(ref_model, ov::test::utils::DEVICE_GPU);
    auto ref_request = ref_compiled.create_infer_request();
    ref_request.set_input_tensor(input);
    ref_request.infer();
    const ov::Tensor ref_output = ref_request.get_output_tensor();

    // Tensor parallel: replicated graph on both GPUs, outputs from rank 0.
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    auto tp_compiled = core.compile_model(tp_model, ov::test::utils::DEVICE_GPU, {ov::hint::model_distribution_policy(tp_policy)});
    auto tp_request = tp_compiled.create_infer_request();
    tp_request.set_input_tensor(input);
    tp_request.infer();
    const ov::Tensor tp_output = tp_request.get_output_tensor();

    ASSERT_EQ(ref_output.get_shape(), tp_output.get_shape());
    ov::test::utils::compare(ref_output, tp_output, 1e-4f, 1e-4f);
}

// [S06] GATE: TPShardTransform must detect the MatMul seam and report it via tp_seams.
// The list must be non-empty, classified and stable across repeated compilations.
TEST_F(TensorParallelCompileGPU, ShardTransformDetectsAndReportsSeams) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    std::shared_ptr<ov::Model> shared_model = model;
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    auto compiled = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props);
    auto seams = compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u) << "Expected exactly one detected MatMul seam";
    // Format: "<kind>:<collective>:<node_type>:<node_name>".
    EXPECT_EQ(seams[0].rfind("N_SPLIT:ALL_GATHER:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];

    // Determinism/stability: recompiling the same model yields an identical seam list.
    auto compiled_again = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU, props);
    auto seams_again = compiled_again.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    EXPECT_EQ(seams, seams_again);

    // The seam list must be advertised among the supported properties.
    auto supported = compiled.get_property(ov::supported_properties).as<std::vector<ov::PropertyName>>();
    EXPECT_NE(std::find(supported.begin(), supported.end(), ov::intel_gpu::tp_seams.name()), supported.end());
}

// [S06] Classification: a projection whose output is added back to the residual stream is
// classified as K_SPLIT (all-reduce(sum)); the mini model above stays N_SPLIT (all-gather).
TEST_F(TensorParallelCompileGPU, ShardTransformClassifiesResidualProjectionAsKSplit) {
    ov::Core core;
    if (count_gpu_devices(core) < 2) {
        GTEST_SKIP() << "Tensor parallel requires at least 2 GPUs";
    }

    auto residual_model = make_residual_projection_model();
    std::set<ov::hint::ModelDistributionPolicy> tp_policy = {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL};
    ov::AnyMap props = {ov::hint::model_distribution_policy(tp_policy)};

    auto compiled = core.compile_model(residual_model, ov::test::utils::DEVICE_GPU, props);
    auto seams = compiled.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    ASSERT_EQ(seams.size(), 1u) << "Expected exactly one detected MatMul seam";
    EXPECT_EQ(seams[0].rfind("K_SPLIT:ALL_REDUCE_SUM:MatMul:", 0), 0u) << "Unexpected seam: " << seams[0];
}

// [S06] Regression: TP-off compilation must not run the pass and must report no seams/ranks.
TEST_F(TensorParallelCompileGPU, DefaultCompileHasNoSeams) {
    ov::Core core;
    std::shared_ptr<ov::Model> shared_model = model;

    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(shared_model, ov::test::utils::DEVICE_GPU));
    auto seams = compiled_model.get_property(ov::intel_gpu::tp_seams).as<std::vector<std::string>>();
    EXPECT_TRUE(seams.empty());
    auto ranks = compiled_model.get_property(ov::intel_gpu::tp_ranks).as<std::vector<int32_t>>();
    EXPECT_TRUE(ranks.empty());
}

}  // namespace
