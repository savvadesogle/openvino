// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// [TP-GPU, S15] Runtime probe for cross-device peer access used by the tensor-parallel transport.
// See gpu_devices_peer_accessible() in device_query.hpp and
// src/plugins/intel_gpu/src/plugin/tp/tp_transport.cpp.

#include "intel_gpu/runtime/device_query.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/zero_api.hpp"

namespace cldnn {

bool gpu_devices_peer_accessible(size_t count) {
    if (count < 2) {
        return false;
    }

    // The probe enumerates the Level Zero devices itself (the same zeDeviceGet order the plugin
    // uses to build m_device_map) and checks every ordered i != j pair. Cached per device count so
    // repeated per-request transport construction does not re-enumerate.
    static std::mutex mutex;
    static std::map<size_t, bool> cache;
    std::lock_guard<std::mutex> lock(mutex);

    auto cached = cache.find(count);
    if (cached != cache.end()) {
        return cached->second;
    }

    bool result = false;
    try {
        // ZeroApi loads ze_loader on construction; when it is missing (or a required core symbol
        // cannot be resolved) the constructor throws and we conservatively fall back to host staging.
        auto ze = ov::ZeroApi::get_instance();
        if (ze && ze->zeDeviceCanAccessPeer != nullptr) {
            uint32_t driver_count = 0;
            if (ze->zeDriverGet(&driver_count, nullptr) == ZE_RESULT_SUCCESS && driver_count > 0) {
                std::vector<ze_driver_handle_t> drivers(driver_count);
                if (ze->zeDriverGet(&driver_count, drivers.data()) == ZE_RESULT_SUCCESS) {
                    std::vector<ze_device_handle_t> devices;
                    for (auto driver : drivers) {
                        uint32_t device_count = 0;
                        if (ze->zeDeviceGet(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS) {
                            continue;
                        }
                        std::vector<ze_device_handle_t> handles(device_count);
                        if (device_count > 0 &&
                            ze->zeDeviceGet(driver, &device_count, handles.data()) == ZE_RESULT_SUCCESS) {
                            devices.insert(devices.end(), handles.begin(), handles.end());
                        }
                    }

                    if (devices.size() >= count) {
                        result = true;
                        for (size_t i = 0; i < count && result; ++i) {
                            for (size_t j = 0; j < count; ++j) {
                                if (i == j) {
                                    continue;
                                }
                                // zeDeviceCanAccessPeer(hDevice, hPeerDevice, pCanAccessPeer).
                                ze_bool_t can_access = 0;
                                const auto status = ze->zeDeviceCanAccessPeer(devices[i], devices[j], &can_access);
                                if (status != ZE_RESULT_SUCCESS || can_access == 0) {
                                    result = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (...) {
        result = false;
    }

    cache[count] = result;
    return result;
}

}  // namespace cldnn
