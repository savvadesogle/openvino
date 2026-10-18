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
//
// [S15] Performance fast-paths:
//   - PEER: the capabilities probe now really queries `zeDeviceCanAccessPeer` across the rank
//     devices (cldnn::gpu_devices_peer_accessible). When every cross pair has peer access the
//     transport reports PEER and routes region copies through an installed device-copy backend
//     (set_peer_region_copy); without a backend (or without peer access, as on 2x A770) it stays
//     on host-USM staging. The branch is prepared but inert on the current hardware.
//   - all-reduce now uses a ring reduce-scatter + all-gather instead of "everyone -> rank0 ->
//     everyone": fp32 accumulation is preserved and the replicated bias/residual term is still
//     counted exactly once. Traffic/critical-path stats are exposed via last_all_reduce_stats().

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
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

//! Probes the inter-rank transport capabilities for a @p num_ranks tensor-parallel group.
//! [S15] Probes `zeDeviceCanAccessPeer` across the first @p num_ranks enumerated Intel GPUs
//! (see cldnn::gpu_devices_peer_accessible): when every cross pair has peer access the
//! capabilities report PEER, otherwise HOST_STAGING. On the target machine (2x A770, Windows;
//! S02/S03 findings, docs/notes/l0-spike.md) peer access is unavailable, so this yields
//! HOST_STAGING. The result is cached per group size; never throws.
const TPTransportCapabilities& tp_transport_capabilities(size_t num_ranks);

//! Which all-reduce algorithm the transport executed for the last collective.
enum class TPAllReduceAlgorithm : uint8_t {
    RING = 0,      //!< ring reduce-scatter + all-gather (S15)
    HOST_SUM = 1,  //!< fallback: read every partial and sum into one fp32 accumulator
};

//! Introspection for the prefill/decode report (bytes/messages moved by the last all-reduce).
struct TPAllReduceStats {
    TPAllReduceAlgorithm algorithm = TPAllReduceAlgorithm::HOST_SUM;
    //! Number of rank-to-rank messages (ring: 2*(k-1)*k chunks; host sum: 0).
    size_t messages = 0;
    //! Payload bytes moved across the inter-rank link by the ring algorithm (0 for host sum).
    size_t bytes = 0;
};

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

    //! Direct device-to-device copy of one axis-aligned region [offset, offset + count) along
    //! @p axis, used by the PEER fast-path instead of the pinned-host round-trip. Returns true on
    //! success; a false/absent backend falls back to host staging. Not installed on the current
    //! platform (peer access unavailable, S02/S03) - the branch is ready for a device-staging
    //! backend where `zeDeviceCanAccessPeer` succeeds.
    using PeerRegionCopy = std::function<bool(const Tensor& src,
                                              const Tensor& dst,
                                              size_t axis,
                                              size_t offset,
                                              size_t count)>;

    //! Probes the platform transport (see tp_transport_capabilities).
    explicit TPTransport(size_t num_ranks);
    //! Uses the supplied capabilities (used by tests / callers that already probed).
    TPTransport(size_t num_ranks, TPTransportCapabilities capabilities);

    size_t num_ranks() const { return m_num_ranks; }
    const TPTransportCapabilities& capabilities() const { return m_capabilities; }

    //! [S15] Installs the PEER copy backend. Only used when `capabilities().kind == PEER`.
    void set_peer_region_copy(PeerRegionCopy backend) { m_peer_region_copy = std::move(backend); }

    //! [S15] Stats (algorithm/messages/bytes) of the most recent all_reduce_sum_fp32().
    const TPAllReduceStats& last_all_reduce_stats() const { return m_all_reduce_stats; }

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
    //! Returns true when the PEER fast-path is usable: peer access was detected AND a copy backend
    //! was installed. Otherwise the collectives use host-USM staging (G0).
    bool use_peer() const { return m_capabilities.kind == TPTransportKind::PEER && static_cast<bool>(m_peer_region_copy); }

    size_t m_num_ranks = 1;
    TPTransportCapabilities m_capabilities;
    //! [S15] Optional direct device-to-device copy backend for the PEER fast-path (unset on this
    //! platform; host staging is used instead).
    PeerRegionCopy m_peer_region_copy;
    //! Reusable fp32 staging accumulator for the fallback host-sum all-reduce (capacity grown on
    //! demand, never shrunk) to avoid per-call allocation churn on the staging path.
    mutable std::vector<float> m_staging;
    //! [S15] Introspection for the last all-reduce collective.
    mutable TPAllReduceStats m_all_reduce_stats;
};

}  // namespace ov::intel_gpu
