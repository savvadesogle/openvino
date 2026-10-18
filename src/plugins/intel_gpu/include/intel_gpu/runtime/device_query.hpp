// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"
#include "engine_configuration.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <algorithm>
#include <vector>

namespace cldnn {

// Fetches all available gpu devices with specific runtime and engine types and (optionally) user context/device handles
struct device_query {
public:
    static int device_id;

    explicit device_query(engine_types engine_type,
                          runtime_types runtime_type,
                          void* user_context = nullptr,
                          void* user_device = nullptr,
                          int ctx_device_id = 0,
                          int target_tile_id = -1,
                          bool initialize_devices = false);

    /// @brief Create device query with default values for engine type and runtime type
    explicit device_query(void* user_context = nullptr,
                          void* user_device = nullptr,
                          int ctx_device_id = 0,
                          int target_tile_id = -1,
                          bool initialize_devices = false);

    std::map<std::string, device::ptr> get_available_devices() const {
        return _available_devices;
    }

    ~device_query() = default;
private:
    std::map<std::string, device::ptr> _available_devices;
};

// One device seen by the lightweight enumeration below.
struct lightweight_device {
    std::string map_id;    ///< Detector's own id for the device (this library's ".N").
    device_info info;      ///< Device info populated without engine/context construction.
};

// Cheap device enumeration for pre-construction dispatch: lists the devices this build's
// runtime can serve with device_info populated, WITHOUT building an engine/context.
std::vector<lightweight_device> lightweight_enumerate() noexcept;

// [TP-GPU, S15] Runtime probe of cross-device peer access for tensor-parallel collectives.
// Returns true iff the first @p count enumerated Intel GPUs can DIRECTLY access each other's
// memory (Level Zero `zeDeviceCanAccessPeer` for every ordered i != j pair). The tensor-parallel
// transport (TPTransport) uses this to select the peer fast-path over host-USM staging (GATE G0).
// Returns false when fewer than two devices are visible, Level Zero is unavailable/resolves no
// peer query, or any cross pair lacks peer access (the common case on 2x Arc A770, S02/S03).
// Result is cached per @p count; never throws.
bool gpu_devices_peer_accessible(size_t count);

}  // namespace cldnn
