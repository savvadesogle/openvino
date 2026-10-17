// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S05/S07/S08/S10] Experimental GPU tensor-parallel support.
// TPInferRequest drives k per-rank inference requests (one per device).
//  - S05: replicated graph, inputs replicated to every rank, rank-0 outputs;
//  - S07: N-split ranks produce only their output-channel slice; rank-0 outputs marked as
//    all-gather are assembled on the host from the per-rank slices (concatenation, no
//    arithmetic);
//  - S08: K-split ranks produce a partial GEMM sum; rank-0 outputs marked as all-reduce are
//    summed in fp32 and the replicated bias/residual terms are added exactly once;
//  - S10: head-sharded state variables are aggregated from the per-rank slices on
//    query_state()/get_state() and scattered back on set_state().
// See PLAN.md §4.3, §4.4, §7.4.

#include "intel_gpu/plugin/tp/tp_infer_request.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/tp/tp_compiled_model.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {

// Copies input data between two host tensors bound to the same port.
// Device (remote) tensors are rejected: replicated inputs are expected to be host-visible,
// which is the case for the default GPU I/O path (see docs/notes/gpu-plugin-arch.md §4).
void copy_host_tensor(const ov::SoPtr<ov::ITensor>& src,
                      const ov::SoPtr<ov::ITensor>& dst,
                      const ov::Output<const ov::Node>& port) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot replicate input tensor for port ", port);
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(src._ptr) == nullptr &&
                        std::dynamic_pointer_cast<ov::IRemoteTensor>(dst._ptr) == nullptr,
                    "[GPU][TP] Input replication for device (remote) tensors is not supported yet (S05)");

    if (port.get_element_type() == ov::element::string) {
        const auto count = dst->get_size();
        auto* s = src->data<std::string>();
        auto* d = dst->data<std::string>();
        for (size_t i = 0; i < count; ++i) {
            d[i] = s[i];
        }
        return;
    }

    const size_t bytes = std::min(src->get_byte_size(), dst->get_byte_size());
    std::memcpy(dst->data(), src->data(), bytes);
}

// [S07/S10] Copies the slice [offset, offset + count) along @p axis of @p src into @p dst at the
// same position - the all-gather concatenation primitive (no arithmetic). @p src must be shaped
// like @p dst with dimension @p axis equal to @p count. Generalizes the S07 last-axis helper so
// head-sharded state outputs (split along an interior axis) assemble the same way.
void copy_axis_region(const ov::SoPtr<ov::ITensor>& src,
                      const ov::SoPtr<ov::ITensor>& dst,
                      size_t axis,
                      size_t offset,
                      size_t count) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot assemble output: null tensor");
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(src._ptr) == nullptr &&
                        std::dynamic_pointer_cast<ov::IRemoteTensor>(dst._ptr) == nullptr,
                    "[GPU][TP] Host assembly of device (remote) tensors is not supported yet (S07)");

    const auto full_shape = dst->get_shape();
    const auto src_shape = src->get_shape();
    OPENVINO_ASSERT(axis < full_shape.size(), "[GPU][TP] Cannot assemble along axis ", axis);
    OPENVINO_ASSERT(src_shape.size() == full_shape.size() && src_shape[axis] == count,
                    "[GPU][TP] Unexpected rank slice shape ", src_shape, " for full shape ", full_shape);
    OPENVINO_ASSERT(offset + count <= full_shape[axis], "[GPU][TP] All-gather region out of bounds");

    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= full_shape[i];
    }
    size_t inner = 1;
    for (size_t i = axis + 1; i < full_shape.size(); ++i) {
        inner *= full_shape[i];
    }
    const size_t axis_dim = full_shape[axis];

    const auto et = dst->get_element_type();
    if (et == ov::element::string) {
        auto* s = src->data<std::string>();
        auto* d = dst->data<std::string>();
        for (size_t o = 0; o < outer; ++o) {
            for (size_t j = 0; j < count; ++j) {
                for (size_t k = 0; k < inner; ++k) {
                    d[(o * axis_dim + offset + j) * inner + k] = s[(o * count + j) * inner + k];
                }
            }
        }
        return;
    }

    const size_t elem_bytes = et.size();
    const size_t slab_bytes = count * inner * elem_bytes;
    const auto* s = static_cast<const uint8_t*>(src->data());
    auto* d = static_cast<uint8_t*>(dst->data());
    for (size_t o = 0; o < outer; ++o) {
        std::memcpy(d + (o * axis_dim + offset) * inner * elem_bytes, s + o * slab_bytes, slab_bytes);
    }
}

// [S10] Inverse of copy_axis_region: copies the region [offset, offset + count) along @p axis of
// the full @p src into the (axis cut to @p count) @p dst. Used to scatter a full state back to the
// per-rank slices on set_state().
void extract_axis_region(const ov::SoPtr<ov::ITensor>& src,
                         const ov::SoPtr<ov::ITensor>& dst,
                         size_t axis,
                         size_t offset,
                         size_t count) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot set state: null tensor");
    const auto full_shape = src->get_shape();
    const auto dst_shape = dst->get_shape();
    OPENVINO_ASSERT(axis < full_shape.size(), "[GPU][TP] Cannot scatter along axis ", axis);
    OPENVINO_ASSERT(dst_shape.size() == full_shape.size() && dst_shape[axis] == count,
                    "[GPU][TP] Unexpected state slice shape ", dst_shape, " for full shape ", full_shape);
    OPENVINO_ASSERT(offset + count <= full_shape[axis], "[GPU][TP] State scatter region out of bounds");

    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= full_shape[i];
    }
    size_t inner = 1;
    for (size_t i = axis + 1; i < full_shape.size(); ++i) {
        inner *= full_shape[i];
    }
    const size_t axis_dim = full_shape[axis];

    const auto et = dst->get_element_type();
    if (et == ov::element::string) {
        auto* s = src->data<std::string>();
        auto* d = dst->data<std::string>();
        for (size_t o = 0; o < outer; ++o) {
            for (size_t j = 0; j < count; ++j) {
                for (size_t k = 0; k < inner; ++k) {
                    d[(o * count + j) * inner + k] = s[(o * axis_dim + offset + j) * inner + k];
                }
            }
        }
        return;
    }

    const size_t elem_bytes = et.size();
    const size_t slab_bytes = count * inner * elem_bytes;
    const auto* s = static_cast<const uint8_t*>(src->data());
    auto* d = static_cast<uint8_t*>(dst->data());
    for (size_t o = 0; o < outer; ++o) {
        std::memcpy(d + o * slab_bytes, s + (o * axis_dim + offset) * inner * elem_bytes, slab_bytes);
    }
}

// [S08] Reads a (host) tensor into an fp32 vector. Device (remote) tensors are rejected.
std::vector<float> tensor_to_floats(const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_ASSERT(tensor, "[GPU][TP] Cannot read a null tensor");
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) == nullptr,
                    "[GPU][TP] Host all-reduce of device (remote) tensors is not supported yet (S08)");
    const size_t count = tensor->get_size();
    std::vector<float> out(count);
    const auto et = tensor->get_element_type();
    if (et == ov::element::f32) {
        const auto* p = tensor->data<float>();
        std::copy(p, p + count, out.begin());
    } else if (et == ov::element::f16) {
        const auto* p = tensor->data<ov::float16>();
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::bf16) {
        const auto* p = tensor->data<ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::f64) {
        const auto* p = tensor->data<double>();
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::i32) {
        const auto* p = tensor->data<int32_t>();
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
    } else if (et == ov::element::i64) {
        const auto* p = tensor->data<int64_t>();
        for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(p[i]);
    } else {
        OPENVINO_THROW("[GPU][TP] Unsupported element type for host all-reduce: ", et);
    }
    return out;
}

// [S08] Writes an fp32 vector back into a (host) tensor, casting to its element type.
void floats_to_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::vector<float>& values) {
    OPENVINO_ASSERT(tensor, "[GPU][TP] Cannot write a null tensor");
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) == nullptr,
                    "[GPU][TP] Host all-reduce of device (remote) tensors is not supported yet (S08)");
    const size_t count = std::min(values.size(), tensor->get_size());
    const auto et = tensor->get_element_type();
    if (et == ov::element::f32) {
        std::copy(values.begin(), values.begin() + count, tensor->data<float>());
    } else if (et == ov::element::f16) {
        auto* p = tensor->data<ov::float16>();
        for (size_t i = 0; i < count; ++i) p[i] = static_cast<ov::float16>(values[i]);
    } else if (et == ov::element::bf16) {
        auto* p = tensor->data<ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) p[i] = static_cast<ov::bfloat16>(values[i]);
    } else if (et == ov::element::f64) {
        auto* p = tensor->data<double>();
        for (size_t i = 0; i < count; ++i) p[i] = static_cast<double>(values[i]);
    } else {
        OPENVINO_THROW("[GPU][TP] Unsupported output element type for host all-reduce: ", et);
    }
}

//! Row-major strides for a shape.
std::vector<size_t> row_major_strides(const ov::Shape& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

// [S08] Accumulates `acc += scale * term` where term (shaped term_shape) is broadcast to
// out_shape by trailing-dimension alignment. Used for replicated residual inputs.
void accumulate_scaled_broadcast(std::vector<float>& acc,
                                 const ov::Shape& out_shape,
                                 const std::vector<float>& term,
                                 const ov::Shape& term_shape,
                                 float scale) {
    const size_t out_rank = out_shape.size();
    OPENVINO_ASSERT(term_shape.size() <= out_rank, "[GPU][TP] Replicated term has higher rank than output");
    ov::Shape aligned(out_rank, 1);
    for (size_t i = 0; i < term_shape.size(); ++i) {
        aligned[out_rank - term_shape.size() + i] = term_shape[i];
    }
    for (size_t i = 0; i < out_rank; ++i) {
        OPENVINO_ASSERT(aligned[i] == out_shape[i] || aligned[i] == 1,
                        "[GPU][TP] Replicated term is not broadcast-compatible with the output");
    }
    const auto term_strides = row_major_strides(aligned);
    const auto out_strides = row_major_strides(out_shape);
    const size_t total = ov::shape_size(out_shape);
    for (size_t idx = 0; idx < total; ++idx) {
        size_t rem = idx;
        size_t tidx = 0;
        for (size_t d = 0; d < out_rank; ++d) {
            const size_t coord = out_strides[d] == 0 ? 0 : rem / out_strides[d];
            rem = out_strides[d] == 0 ? 0 : rem % out_strides[d];
            if (aligned[d] != 1) {
                tidx += coord * term_strides[d];
            }
        }
        acc[idx] += scale * term[tidx];
    }
}

// [S10] Aggregates one logical state variable across the k ranks. For HEAD_SHARDED variables the
// per-rank states hold disjoint head-axis slices, so get_state() concatenates them into the full
// state and set_state() scatters a full state back into the per-rank slices. REPLICATED variables
// (the common case - dynamic / non-head-local chains) are rank-0 authoritative: get_state()
// returns rank 0 and reset()/set_state() fan out to every rank so ranks stay in sync.
class TPAggregatedState : public ov::IVariableState {
public:
    TPAggregatedState(const std::string& name,
                      std::vector<ov::SoPtr<ov::IVariableState>> ranks,
                      ov::intel_gpu::TPStateShardKind kind,
                      int64_t axis,
                      ov::Shape full_shape)
        : ov::IVariableState(name),
          m_ranks(std::move(ranks)),
          m_kind(kind),
          m_axis(axis),
          m_full_shape(std::move(full_shape)) {}

    void reset() override {
        for (auto& s : m_ranks) {
            if (s) {
                s->reset();
            }
        }
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state, "[GPU][TP] Cannot set a null state tensor for ", m_name);
        if (m_kind != ov::intel_gpu::TPStateShardKind::HEAD_SHARDED || m_ranks.size() <= 1) {
            for (auto& s : m_ranks) {
                if (s) {
                    s->set_state(state);
                }
            }
            return;
        }
        const size_t axis = static_cast<size_t>(m_axis);
        OPENVINO_ASSERT(axis < m_full_shape.size(), "[GPU][TP] Invalid head axis for state ", m_name);
        const size_t count = m_full_shape[axis] / m_ranks.size();
        for (size_t r = 0; r < m_ranks.size(); ++r) {
            if (!m_ranks[r]) {
                continue;
            }
            auto part_shape = m_full_shape;
            part_shape[axis] = count;
            auto slice = ov::SoPtr<ov::ITensor>(ov::make_tensor(state->get_element_type(), part_shape), nullptr);
            extract_axis_region(state, slice, axis, r * count, count);
            m_ranks[r]->set_state(slice);
        }
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        if (m_kind != ov::intel_gpu::TPStateShardKind::HEAD_SHARDED || m_ranks.size() <= 1) {
            return m_ranks.front() ? m_ranks.front()->get_state() : ov::SoPtr<ov::ITensor>{};
        }
        auto first = m_ranks.front()->get_state();
        OPENVINO_ASSERT(first, "[GPU][TP] Rank 0 returned a null state for ", m_name);
        auto full = ov::SoPtr<ov::ITensor>(ov::make_tensor(first->get_element_type(), m_full_shape), nullptr);
        const size_t axis = static_cast<size_t>(m_axis);
        const size_t count = m_full_shape[axis] / m_ranks.size();
        for (size_t r = 0; r < m_ranks.size(); ++r) {
            if (!m_ranks[r]) {
                continue;
            }
            auto part = m_ranks[r]->get_state();
            copy_axis_region(part, full, axis, r * count, count);
        }
        return full;
    }

private:
    std::vector<ov::SoPtr<ov::IVariableState>> m_ranks;
    ov::intel_gpu::TPStateShardKind m_kind;
    int64_t m_axis;
    ov::Shape m_full_shape;
};

}  // namespace

namespace ov::intel_gpu {

TPInferRequest::TPInferRequest(const std::shared_ptr<const TPCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model, "[GPU][TP] TPInferRequest requires a tensor-parallel compiled model");
    const auto& rank_models = compiled_model->get_rank_models();
    OPENVINO_ASSERT(!rank_models.empty(), "[GPU][TP] Tensor-parallel compiled model has no rank models");

    m_rank_requests.reserve(rank_models.size());
    for (const auto& rank_model : rank_models) {
        OPENVINO_ASSERT(rank_model, "[GPU][TP] Encountered a null rank model");
        m_rank_requests.push_back(rank_model->create_sync_infer_request());
    }
    m_output_tensors.resize(get_outputs().size());
}

void TPInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "TPInferRequest::infer");

    // Ranks receive replicated inputs: make sure each rank sees the same data as rank 0.
    replicate_inputs();

    // MVP: run ranks sequentially. Concurrent / device-side seam execution is S08+.
    for (auto& request : m_rank_requests) {
        request->infer();
    }

    // Gather the per-rank N-slices into the full outputs on rank 0.
    assemble_outputs();
}

void TPInferRequest::replicate_inputs() {
    const auto& inputs = get_inputs();
    auto& master = m_rank_requests.front();
    for (const auto& port : inputs) {
        auto src = master->get_tensor(port);
        for (size_t r = 1; r < m_rank_requests.size(); ++r) {
            auto dst = m_rank_requests[r]->get_tensor(port);
            // When the user bound one shared tensor to every rank (default set_tensor path)
            // the pointers coincide and no copy is needed.
            if (src._ptr == dst._ptr) {
                continue;
            }
            copy_host_tensor(src, dst, port);
        }
    }
}

ov::SoPtr<ov::ITensor>& TPInferRequest::output_tensor(size_t index) const {
    if (m_output_tensors.size() <= index) {
        m_output_tensors.resize(index + 1);
    }
    auto& tensor = m_output_tensors[index];
    if (!tensor) {
        const auto& port = get_outputs().at(index);
        tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(port.get_element_type(), port.get_shape()), nullptr);
    }
    return tensor;
}

void TPInferRequest::assemble_outputs() {
    const auto compiled = std::static_pointer_cast<const TPCompiledModel>(get_compiled_model());
    const auto& plan = compiled->get_tp_plan();
    const size_t num_outputs = get_outputs().size();
    const size_t num_ranks = m_rank_requests.size();

    for (size_t i = 0; i < num_outputs; ++i) {
        auto& dst = output_tensor(i);
        const auto* seam = plan.find_output_seam(i);
        const bool sharded = seam != nullptr && num_ranks > 1;
        if (sharded && seam->collective == TPCollectiveKind::ALL_REDUCE_SUM) {
            // [S08] K-split: sum the per-rank partial GEMM outputs (fp32, all-reduce sum).
            assemble_reduce_sum(i, *seam);
        } else if (sharded && seam->collective == TPCollectiveKind::ALL_GATHER && seam->shard_dim_size > 0) {
            // All-gather: concatenate each rank's slice along the split axis (last axis for
            // N-split GEMM outputs, an interior head axis for head-sharded state outputs, S10).
            const auto full_shape = dst->get_shape();
            OPENVINO_ASSERT(!full_shape.empty(), "[GPU][TP] Cannot assemble a scalar output");
            const int64_t full_rank = static_cast<int64_t>(full_shape.size());
            const int64_t axis_i = seam->split_axis < 0 ? full_rank + seam->split_axis : seam->split_axis;
            OPENVINO_ASSERT(axis_i >= 0 && axis_i < full_rank, "[GPU][TP] Invalid all-gather axis ", seam->split_axis);
            const size_t axis = static_cast<size_t>(axis_i);
            const size_t count = full_shape[axis] / num_ranks;
            for (size_t r = 0; r < num_ranks; ++r) {
                auto rank_port = m_rank_requests[r]->get_outputs().at(i);
                auto src = m_rank_requests[r]->get_tensor(rank_port);
                copy_axis_region(src, dst, axis, r * count, count);
            }
        } else {
            // Replicated output: rank 0 is authoritative.
            auto rank_port = m_rank_requests.front()->get_outputs().at(i);
            auto src = m_rank_requests.front()->get_tensor(rank_port);
            copy_host_tensor(src, dst, get_outputs().at(i));
        }
    }
}

void TPInferRequest::assemble_reduce_sum(size_t output_index, const TPOutputSeam& seam) {
    auto& dst = output_tensor(output_index);
    const size_t num_ranks = m_rank_requests.size();
    const size_t out_size = dst->get_size();
    const auto out_shape = dst->get_shape();

    // 1) Sum every rank's partial GEMM output in fp32 (PLAN §7.4: no fp16 summation).
    std::vector<float> acc(out_size, 0.0f);
    for (size_t r = 0; r < num_ranks; ++r) {
        auto rank_port = m_rank_requests[r]->get_outputs().at(output_index);
        auto src = m_rank_requests[r]->get_tensor(rank_port);
        auto partial = tensor_to_floats(src);
        OPENVINO_ASSERT(partial.size() == out_size,
                        "[GPU][TP] K-split partial output shape mismatch on rank ", r);
        for (size_t j = 0; j < out_size; ++j) {
            acc[j] += partial[j];
        }
    }

    // 2) Replicated bias/residual terms are present once per rank: subtract the (num_ranks - 1)
    //    extra copies so the term is counted exactly once (PLAN §4.3).
    const float neg_k1 = -static_cast<float>(num_ranks - 1);
    if (num_ranks > 1) {
        for (const auto& add : seam.replicated_adds) {
            if (add.kind == TPReplicatedAddKind::CONSTANT) {
                OPENVINO_ASSERT(add.values.size() == out_size,
                                "[GPU][TP] K-split replicated constant size mismatch");
                for (size_t j = 0; j < out_size; ++j) {
                    acc[j] += neg_k1 * add.values[j];
                }
            } else if (add.kind == TPReplicatedAddKind::INPUT) {
                const auto& in_port = get_inputs().at(add.input_index);
                auto t = m_rank_requests.front()->get_tensor(in_port);
                auto values = tensor_to_floats(t);
                accumulate_scaled_broadcast(acc, out_shape, values, t->get_shape(), neg_k1);
            }
        }
    }

    // 3) Cast back to the output element type (single conversion after fp32 accumulation).
    floats_to_tensor(dst, acc);
}

ov::SoPtr<ov::ITensor> TPInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);
    if (found_port.is_input()) {
        // The "master" view of inputs lives on rank 0.
        return m_rank_requests.front()->get_tensor(port);
    }
    // Outputs are served from the (possibly assembled) full-shape buffers owned here.
    return output_tensor(found_port.idx);
}

void TPInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);

    // Base-class bookkeeping + validation (element type / shape checks).
    ov::ISyncInferRequest::set_tensor(port, tensor);

    if (found_port.is_input()) {
        // Replicate the input on every rank. A host tensor is safe to share across devices
        // (each rank copies it into its own device memory during enqueue).
        for (auto& request : m_rank_requests) {
            request->set_tensor(port, tensor);
        }
    } else {
        // Outputs are owned by this request and filled during infer().
        output_tensor(found_port.idx) = tensor;
    }
}

std::vector<ov::SoPtr<ov::ITensor>> TPInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    auto found_port = find_port(port);
    OPENVINO_ASSERT(found_port.found(), "[GPU][TP] Cannot find tensor for port ", port);
    if (found_port.is_input()) {
        return m_rank_requests.front()->get_tensors(port);
    }
    return {get_tensor(port)};
}

void TPInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                 const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    if (tensors.size() == 1) {
        set_tensor(port, tensors[0]);
        return;
    }
    // Batched (per-item) inputs are not part of the S05 scope.
    OPENVINO_THROW_NOT_IMPLEMENTED("[GPU][TP] Batched input tensors are not supported by the tensor-parallel request yet");
}

void TPInferRequest::check_tensors() const {
    // Tensors are owned by the per-rank requests and validated there; rank 0 holds the
    // authoritative input views and the assembled outputs are validated on set/alloc.
    return;
}

std::vector<ov::ProfilingInfo> TPInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t r = 0; r < m_rank_requests.size(); ++r) {
        auto rank_info = m_rank_requests[r]->get_profiling_info();
        for (auto& rec : rank_info) {
            rec.node_name = "rank" + std::to_string(r) + ": " + rec.node_name;
        }
        info.insert(info.end(), rank_info.begin(), rank_info.end());
    }
    return info;
}

std::vector<ov::SoPtr<ov::IVariableState>> TPInferRequest::query_state() const {
    // [S10] Aggregate the per-rank state variables. HEAD_SHARDED variables are concatenated along
    // their head axis; REPLICATED variables are rank-0 authoritative but reset/set_state fan out.
    const auto compiled = std::static_pointer_cast<const TPCompiledModel>(get_compiled_model());
    const auto& plan = compiled->get_tp_plan();
    const auto num_ranks = m_rank_requests.size();

    std::vector<std::vector<ov::SoPtr<ov::IVariableState>>> all_states;
    all_states.reserve(num_ranks);
    for (const auto& request : m_rank_requests) {
        all_states.push_back(request->query_state());
    }
    const auto& rank0_states = all_states.front();

    std::vector<ov::SoPtr<ov::IVariableState>> out;
    out.reserve(rank0_states.size());
    for (size_t i = 0; i < rank0_states.size(); ++i) {
        if (!rank0_states[i]) {
            continue;
        }
        const std::string name = rank0_states[i]->get_name();
        std::vector<ov::SoPtr<ov::IVariableState>> per_rank;
        per_rank.reserve(num_ranks);
        for (const auto& states : all_states) {
            ov::SoPtr<ov::IVariableState> found;
            for (const auto& s : states) {
                if (s && s->get_name() == name) {
                    found = s;
                    break;
                }
            }
            if (!found && states.size() == rank0_states.size()) {
                found = states[i];
            }
            per_rank.push_back(found);
        }
        if (per_rank.empty() || !per_rank.front()) {
            per_rank.assign(num_ranks, rank0_states[i]);
        }

        const auto* seam = plan.find_state_seam(name);
        if (seam != nullptr && seam->kind == TPStateShardKind::HEAD_SHARDED && num_ranks > 1) {
            out.emplace_back(std::make_shared<TPAggregatedState>(name,
                                                                 std::move(per_rank),
                                                                 TPStateShardKind::HEAD_SHARDED,
                                                                 seam->axis,
                                                                 seam->full_shape),
                             nullptr);
        } else {
            out.emplace_back(std::make_shared<TPAggregatedState>(name,
                                                                 std::move(per_rank),
                                                                 TPStateShardKind::REPLICATED,
                                                                 -1,
                                                                 ov::Shape{}),
                             nullptr);
        }
    }
    return out;
}

}  // namespace ov::intel_gpu
