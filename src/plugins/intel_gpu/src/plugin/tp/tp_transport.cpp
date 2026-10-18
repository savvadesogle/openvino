// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S12/S15] Experimental GPU tensor-parallel support.
// Inter-rank transport for collectives. S12: host-USM staging (GATE G0: the only working
// inter-device path on 2x Arc A770, docs/notes/l0-spike.md §8). S15: real `zeDeviceCanAccessPeer`
// probe selecting the PEER fast-path when available, and a ring reduce-scatter + all-gather
// all-reduce (fp32, replicated terms once) instead of the previous "everyone -> rank0 -> everyone"
// host sum. See tp_transport.hpp and docs/notes/s15-perf-report.md.

#include "intel_gpu/plugin/tp/tp_transport.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "intel_gpu/runtime/device_query.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::intel_gpu {

namespace {

bool is_remote(const ov::SoPtr<ov::ITensor>& t) {
    return t && std::dynamic_pointer_cast<ov::IRemoteTensor>(t._ptr) != nullptr;
}

//! Copies the slice [offset, offset + count) along @p axis of @p src into @p dst at the same
//! position - the all-gather concatenation primitive (no arithmetic). @p src must be shaped like
//! @p dst with dimension @p axis equal to @p count.
void copy_axis_region(const ov::SoPtr<ov::ITensor>& src,
                      const ov::SoPtr<ov::ITensor>& dst,
                      size_t axis,
                      size_t offset,
                      size_t count) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot assemble tensor: null tensor");
    OPENVINO_ASSERT(!is_remote(src) && !is_remote(dst),
                    "[GPU][TP] Host staging of device (remote) tensors is not supported "
                    "(host-USM staging is the G0 transport, see docs/notes/l0-spike.md §8)");

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

//! Inverse of copy_axis_region: copies the region [offset, offset + count) along @p axis of the
//! full @p src into the (axis cut to @p count) @p dst. Used to scatter a full state back to the
//! per-rank slices on set_state().
void extract_axis_region(const ov::SoPtr<ov::ITensor>& src,
                         const ov::SoPtr<ov::ITensor>& dst,
                         size_t axis,
                         size_t offset,
                         size_t count) {
    OPENVINO_ASSERT(src && dst, "[GPU][TP] Cannot scatter tensor: null tensor");
    OPENVINO_ASSERT(!is_remote(src) && !is_remote(dst),
                    "[GPU][TP] Host staging of device (remote) tensors is not supported");
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

//! Reads a (host) tensor into an fp32 vector. Device (remote) tensors are rejected.
std::vector<float> tensor_to_floats(const ov::SoPtr<ov::ITensor>& tensor) {
    OPENVINO_ASSERT(tensor, "[GPU][TP] Cannot read a null tensor");
    OPENVINO_ASSERT(!is_remote(tensor),
                    "[GPU][TP] Host all-reduce of device (remote) tensors is not supported at the staging "
                    "boundary");
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

//! Writes an fp32 vector back into a (host) tensor, casting to its element type.
void floats_to_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::vector<float>& values) {
    OPENVINO_ASSERT(tensor, "[GPU][TP] Cannot write a null tensor");
    OPENVINO_ASSERT(!is_remote(tensor),
                    "[GPU][TP] Host all-reduce of device (remote) tensors is not supported at the staging "
                    "boundary");
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

//! Accumulates `acc += scale * term` where term (shaped @p term_shape) is broadcast to @p
//! out_shape by trailing-dimension alignment. Used for replicated residual inputs.
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

//! [S15] Ring all-reduce(sum) over the per-rank fp32 partials: reduce-scatter (k-1 steps; each rank
//! forwards one chunk to its successor and accumulates the chunk received from its predecessor in
//! fp32) followed by all-gather (k-1 steps). Afterwards every rank holds the full sum; rank 0's
//! buffer is returned. Requires S divisible by k (callers fall back to the host sum otherwise).
//! Payload moved per rank = 2*(k-1)/k*S, i.e. for k=2 a single S-element arm in each direction,
//! halving the critical-path latency of the previous "everyone -> rank0 -> everyone" scheme (2S).
void ring_all_reduce_fp32(const std::vector<std::vector<float>>& partials,
                          std::vector<float>& out,
                          size_t& messages,
                          size_t& bytes) {
    const size_t k = partials.size();
    const size_t elements = partials.front().size();
    const size_t chunk = elements / k;
    std::vector<std::vector<float>> buffers = partials;  // each rank's working buffer

    messages = 0;
    bytes = 0;
    std::vector<float> msg(k * chunk);

    // Reduce-scatter: after k-1 steps rank r owns the fully reduced chunk (r + 1) % k.
    for (size_t step = 0; step + 1 < k; ++step) {
        for (size_t r = 0; r < k; ++r) {
            const size_t idx = (r + k - step) % k;
            std::copy_n(buffers[r].data() + idx * chunk, chunk, msg.data() + r * chunk);
        }
        for (size_t r = 0; r < k; ++r) {
            const size_t src = (r + k - 1) % k;
            const size_t recv_idx = (r + k - step - 1) % k;
            float* dst = buffers[r].data() + recv_idx * chunk;
            const float* incoming = msg.data() + src * chunk;
            for (size_t j = 0; j < chunk; ++j) {
                dst[j] += incoming[j];
            }
        }
        messages += k;
        bytes += k * chunk * sizeof(float);
    }

    // All-gather: rank r forwards the chunk it currently owns; after k-1 steps every chunk is known.
    for (size_t step = 0; step + 1 < k; ++step) {
        for (size_t r = 0; r < k; ++r) {
            const size_t idx = (r + k + 1 - step) % k;
            std::copy_n(buffers[r].data() + idx * chunk, chunk, msg.data() + r * chunk);
        }
        for (size_t r = 0; r < k; ++r) {
            const size_t src = (r + k - 1) % k;
            const size_t idx = (src + k + 1 - step) % k;
            std::copy_n(msg.data() + src * chunk, chunk, buffers[r].data() + idx * chunk);
        }
        messages += k;
        bytes += k * chunk * sizeof(float);
    }

    out = std::move(buffers.front());  // rank 0 holds the full sum
}

}  // namespace

const TPTransportCapabilities& tp_transport_capabilities(size_t num_ranks) {
    // [S15] Probes the inter-rank transport once per group size. `gpu_devices_peer_accessible`
    // queries Level Zero `zeDeviceCanAccessPeer` across the first `num_ranks` enumerated Intel
    // GPUs (never throws). On the target machine (2x Arc A770, Windows, L0 1.33.1) peer access is
    // false across devices, IPC memory is UNSUPPORTED_FEATURE and IPC events are unavailable, so
    // the only working inter-device path is host-USM staging (GATE G0); the result is cached and
    // is deliberately conservative (a true PEER result requires every cross pair to have access).
    static std::mutex mutex;
    static std::map<size_t, TPTransportCapabilities> cache;
    std::lock_guard<std::mutex> lock(mutex);

    auto cached = cache.find(num_ranks);
    if (cached != cache.end()) {
        return cached->second;
    }

    TPTransportCapabilities caps;
    caps.peer_available = cldnn::gpu_devices_peer_accessible(num_ranks);
    caps.ipc_available = false;  // S02: IPC memory/events are UNSUPPORTED_FEATURE on Windows
    caps.kind = caps.peer_available ? TPTransportKind::PEER : TPTransportKind::HOST_STAGING;
    caps.reason = caps.peer_available
                      ? "peer fast-path: zeDeviceCanAccessPeer across all rank pairs (S15)"
                      : "host-USM staging (G0): peer access and IPC memory/events unavailable on this platform (S02/S03)";
    return cache.emplace(num_ranks, std::move(caps)).first->second;
}

TPTransport::TPTransport(size_t num_ranks) : TPTransport(num_ranks, tp_transport_capabilities(num_ranks)) {}

TPTransport::TPTransport(size_t num_ranks, TPTransportCapabilities capabilities)
    : m_num_ranks(std::max<size_t>(1, num_ranks)),
      m_capabilities(std::move(capabilities)) {}

void TPTransport::all_gather_axis(const std::vector<Tensor>& slices, const Tensor& full, size_t axis) const {
    OPENVINO_ASSERT(full, "[GPU][TP] all_gather: null full tensor");
    OPENVINO_ASSERT(!slices.empty(), "[GPU][TP] all_gather: no rank slices");
    const auto full_shape = full->get_shape();
    OPENVINO_ASSERT(axis < full_shape.size(), "[GPU][TP] all_gather: axis ", axis, " out of range");
    OPENVINO_ASSERT(full_shape[axis] % slices.size() == 0,
                    "[GPU][TP] all_gather: axis ", axis, " dim ", full_shape[axis], " not divisible by ",
                    slices.size(), " ranks");
    const size_t count = full_shape[axis] / slices.size();
    for (size_t r = 0; r < slices.size(); ++r) {
        // [S15] PEER fast-path (direct device copy) when available; host staging otherwise (G0).
        if (use_peer() && m_peer_region_copy(slices[r], full, axis, r * count, count)) {
            continue;
        }
        copy_axis_region(slices[r], full, axis, r * count, count);
    }
}

void TPTransport::scatter_axis(const Tensor& full, size_t axis, const std::vector<Tensor>& slices) const {
    OPENVINO_ASSERT(full, "[GPU][TP] scatter: null full tensor");
    OPENVINO_ASSERT(!slices.empty(), "[GPU][TP] scatter: no rank slices");
    const auto full_shape = full->get_shape();
    OPENVINO_ASSERT(axis < full_shape.size(), "[GPU][TP] scatter: axis ", axis, " out of range");
    OPENVINO_ASSERT(full_shape[axis] % slices.size() == 0,
                    "[GPU][TP] scatter: axis ", axis, " dim ", full_shape[axis], " not divisible by ",
                    slices.size(), " ranks");
    const size_t count = full_shape[axis] / slices.size();
    for (size_t r = 0; r < slices.size(); ++r) {
        // [S15] PEER fast-path (direct device copy) when available; host staging otherwise (G0).
        if (use_peer() && m_peer_region_copy(full, slices[r], axis, r * count, count)) {
            continue;
        }
        extract_axis_region(full, slices[r], axis, r * count, count);
    }
}

void TPTransport::all_reduce_sum_fp32(const std::vector<Tensor>& partials,
                                      const Tensor& full,
                                      const std::vector<TPReplicatedAdd>& replicated_adds,
                                      const std::vector<Tensor>& add_inputs) const {
    OPENVINO_ASSERT(full, "[GPU][TP] all_reduce: null output tensor");
    OPENVINO_ASSERT(!partials.empty(), "[GPU][TP] all_reduce: no rank partials");
    const size_t out_size = full->get_size();
    const auto out_shape = full->get_shape();

    // 1) Reduce every rank's partial GEMM output in fp32 (PLAN §7.4: no fp16 summation). [S15]
    //    Ring reduce-scatter + all-gather when the element count is divisible by the rank count
    //    (the standard case); otherwise fall back to the previous host sum into the reusable
    //    staging accumulator. Both keep full fp32 accumulation.
    const size_t num_ranks = partials.size();
    // Reuse the transport-owned staging accumulator (grown on demand, never shrunk).
    std::vector<float>& acc = m_staging;
    if (num_ranks > 1 && out_size % num_ranks == 0) {
        std::vector<std::vector<float>> buffers;
        buffers.reserve(num_ranks);
        for (size_t r = 0; r < num_ranks; ++r) {
            buffers.push_back(tensor_to_floats(partials[r]));
            OPENVINO_ASSERT(buffers.back().size() == out_size,
                            "[GPU][TP] K-split partial output shape mismatch on rank ", r);
        }
        std::vector<float> reduced;
        ring_all_reduce_fp32(buffers, reduced, m_all_reduce_stats.messages, m_all_reduce_stats.bytes);
        acc = std::move(reduced);
        m_all_reduce_stats.algorithm = TPAllReduceAlgorithm::RING;
    } else {
        acc.assign(out_size, 0.0f);
        for (size_t r = 0; r < num_ranks; ++r) {
            auto partial = tensor_to_floats(partials[r]);
            OPENVINO_ASSERT(partial.size() == out_size,
                            "[GPU][TP] K-split partial output shape mismatch on rank ", r);
            for (size_t j = 0; j < out_size; ++j) {
                acc[j] += partial[j];
            }
        }
        m_all_reduce_stats.algorithm = TPAllReduceAlgorithm::HOST_SUM;
        m_all_reduce_stats.messages = 0;
        m_all_reduce_stats.bytes = 0;
    }

    // 2) Replicated bias/residual terms are present once per rank: subtract the (num_ranks - 1)
    //    extra copies so the term is counted exactly once (PLAN §4.3).
    const float neg_k1 = -static_cast<float>(partials.size() - 1);
    if (partials.size() > 1) {
        for (size_t i = 0; i < replicated_adds.size(); ++i) {
            const auto& add = replicated_adds[i];
            if (add.kind == TPReplicatedAddKind::CONSTANT) {
                OPENVINO_ASSERT(add.values.size() == out_size,
                                "[GPU][TP] K-split replicated constant size mismatch");
                for (size_t j = 0; j < out_size; ++j) {
                    acc[j] += neg_k1 * add.values[j];
                }
            } else if (add.kind == TPReplicatedAddKind::INPUT) {
                OPENVINO_ASSERT(i < add_inputs.size() && add_inputs[i],
                                "[GPU][TP] K-split replicated input tensor was not resolved");
                auto values = tensor_to_floats(add_inputs[i]);
                accumulate_scaled_broadcast(acc, out_shape, values, add_inputs[i]->get_shape(), neg_k1);
            }
        }
    }

    // 3) Cast back to the output element type (single conversion after fp32 accumulation).
    floats_to_tensor(full, acc);
}

}  // namespace ov::intel_gpu
