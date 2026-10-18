// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S12] Experimental GPU tensor-parallel support.
// Inter-rank transport for tensor-parallel collectives (all-gather / all-reduce, PLAN §4.4).
//
// The only working inter-device path on the target machine (2x Intel Arc A770, Windows, L0
// 1.33.1) is host-USM staging: device_i -> pinned host -> device_j, ~4-5 GB/s end-to-end
// (GATE G0, S03, docs/notes/l0-spike.md §8). Peer access (`zeDeviceCanAccessPeer`) and IPC
// memory/events are unavailable there (S02), so `TPTransport` defaults to host staging and
// exposes the peer path as a runtime fast-path hook only.
//
// TPTransport encapsulates the host boundary collectives used to assemble tensor-parallel
// outputs and aggregate state variables:
//   - all_gather: concatenate the per-rank slices of a sharded axis (N-split / head-sharded);
//   - scatter:    inverse of all_gather (distribute a full state back to per-rank slices);
//   - all_reduce: fp32 all-reduce(sum) of partial GEMM outputs, adding replicated bias/residual
//                 terms exactly once (PLAN §4.3, §7.4).
// It owns a reusable pinned-host staging buffer so the collective path does not allocate per
// call. The device->host->device copies are performed by the per-rank requests (their tensors
// are host-visible); this class owns the staging/sync boundary between them.
//
// Device-side events (`zeEventPool`, peer copies) and true in-graph inter-node collectives
// (splitting the model between ranks) are NOT part of S12 - see STATE.md S12 notes.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "intel_gpu/plugin/tp/tp_plan.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
class ITensor;
}  // namespace ov

namespace ov::intel_gpu {

//! How tensor-parallel collectives are transported between ranks.
enum class TPTransportKind : uint8_t {
    HOST_STAGING = 0,  //!< device_i -> pinned host -> device_j (G0 default; works everywhere)
    PEER = 1,          //!< direct device-to-device / peer copies (runtime fast-path; S15)
};

//! Result of probing the inter-rank transport capabilities at compile time.
struct TPTransportCapabilities {
    //! Transport selected for this TP group.
    TPTransportKind kind = TPTransportKind::HOST_STAGING;
    //! Whether cross-device peer access is available (`zeDeviceCanAccessPeer`).
    bool peer_available = false;
    //! Whether IPC memory / IPC events are usable across devices on this platform.
    bool ipc_available = false;
    //! Human-readable explanation of the probe result (logged once).
    std::string reason;
};

//! Probes the inter-rank transport capabilities. On the target machine (2xA770, Windows) this
//! always yields HOST_STAGING with peer/ipc disabled (S02/S03 findings, docs/notes/l0-spike.md).
//! The peer check is a runtime fast-path hook: when `zeDeviceCanAccessPeer` returns true the
//! transport may switch to PEER (not exercised on the current hardware).
const TPTransportCapabilities& tp_transport_capabilities();

/**
 * @brief Host-USM staging transport for inter-rank collectives.
 *
 * All tensors passed to the collectives are host-visible (`ov::ITensor`); remote/device tensors
 * are rejected, matching the G0 staging contract (per-rank requests expose host tensors). The
 * transport owns a reusable pinned-host staging buffer.
 */
class TPTransport {
public:
    using Tensor = ov::SoPtr<ov::ITensor>;

    explicit TPTransport(size_t num_ranks);

    size_t num_ranks() const { return m_num_ranks; }
    const TPTransportCapabilities& capabilities() const { return m_capabilities; }

    /**
     * @brief Concatenates per-rank slices into the full tensor along @p axis (all-gather).
     *
     * @p slices[r] must have the same rank/shape as @p full except along @p axis where it holds
     * `full[axis] / num_ranks` elements; rank r's slice is placed at offset `r * count`.
     */
    void all_gather_axis(const std::vector<Tensor>& slices, const Tensor& full, size_t axis) const;

    /**
     * @brief Inverse of all_gather_axis: copies the per-rank slices out of a full tensor.
     * Used to scatter a full state variable back to the per-rank shards.
     */
    void scatter_axis(const Tensor& full, size_t axis, const std::vector<Tensor>& slices) const;

    /**
     * @brief fp32 all-reduce(sum) of per-rank partial outputs into @p full, then adds the
     * replicated additive terms exactly once (PLAN §4.3, §7.4).
     *
     * @p replicated_adds are the bias/residual terms folded into every rank's partial; the
     * reducer subtracts `(num_ranks - 1) * term` so each term is counted exactly once.
     * @p add_inputs is parallel to @p replicated_adds; for INPUT-kind terms it holds the resolved
     * replicated input tensor (a null entry is expected for CONSTANT-kind terms).
     */
    void all_reduce_sum_fp32(const std::vector<Tensor>& partials,
                             const Tensor& full,
                             const std::vector<TPReplicatedAdd>& replicated_adds,
                             const std::vector<Tensor>& add_inputs) const;

private:
    size_t m_num_ranks = 1;
    TPTransportCapabilities m_capabilities;
    //! Reusable fp32 staging accumulator for all-reduce (capacity grown on demand, never shrunk)
    //! to avoid per-call allocation churn on the staging path.
    mutable std::vector<float> m_staging;
};

}  // namespace ov::intel_gpu
