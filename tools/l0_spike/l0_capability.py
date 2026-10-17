#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# S02 L0 capability spike (read-only).
#
# Enumerates Level Zero GPU drivers/devices on Windows via ctypes over
# ze_loader.dll and probes inter-device capabilities needed for TP collectives:
#   * zeDeviceCanAccessPeer matrix (N x N, incl. i == j)
#   * zeDeviceGetP2PProperties (ACCESS / ATOMICS) for accessible pairs
#   * IPC memory: zeMemAllocDevice -> zeMemGetIpcHandle -> zeMemOpenIpcHandle
#   * IPC counter-based events (zex): create/get-ipc/open-ipc + host smoke
#
# No performance measurements here (that is S03). Struct layouts were transcribed
# from the system Level Zero header (LevelZeroSDK 1.33.1, ze_api.h) plus the
# bundled zex headers (src/plugins/intel_gpu/src/runtime/ze/compute_runtime/).
#
# Usage:  python l0_capability.py
#
# It does not modify anything; it only allocates 1 MiB device USM per GPU and
# frees it before exit.

import ctypes
from ctypes import (POINTER, Structure, byref, sizeof, cast,
                    c_char, c_char_p, c_int32, c_int64, c_uint8, c_uint32,
                    c_uint64, c_void_p, CFUNCTYPE)

# ---------------------------------------------------------------------------
# Level Zero basic types / enums (values from ze_api.h 1.33.1)
# ---------------------------------------------------------------------------
ze_result_t = c_int32
ze_bool_t = c_uint8
ze_structure_type_t = c_uint32
ze_structure_type_ext_t = c_uint32
ze_handle = c_void_p

ZE_RESULT_SUCCESS = 0

ZE_BIT = lambda i: (1 << i)
ZE_INIT_FLAG_GPU_ONLY = ZE_BIT(0)  # 0x1

ZE_DEVICE_TYPE_GPU = 1

# structure types
ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1
ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES = 0x2
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x4
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 0x7
ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 0xB
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 0x10008

# event scope flags
ZE_EVENT_SCOPE_FLAG_SUBDEVICE = ZE_BIT(0)
ZE_EVENT_SCOPE_FLAG_DEVICE = ZE_BIT(1)
ZE_EVENT_SCOPE_FLAG_HOST = ZE_BIT(2)

# p2p property flags
ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = ZE_BIT(0)
ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = ZE_BIT(1)

# ipc property flags
ZE_IPC_PROPERTY_FLAG_MEMORY = ZE_BIT(0)
ZE_IPC_PROPERTY_FLAG_EVENT_POOL = ZE_BIT(1)

# zex counter-based event
ZEX_BIT = lambda i: (1 << i)
ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC = 0x0003001C
ZEX_STRUCTURE_COUNTER_BASED_EVENT_EXTERNAL_SYNC_ALLOC_PROPERTIES = 0x0003001D
ZEX_STRUCTURE_COUNTER_BASED_EVENT_EXTERNAL_STORAGE_ALLOC_PROPERTIES = 0x00030027

ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE = ZEX_BIT(0)
ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE = ZEX_BIT(1)
ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE = ZEX_BIT(2)
ZEX_COUNTER_BASED_EVENT_FLAG_IPC = ZEX_BIT(3)

ZE_MAX_IPC_HANDLE_SIZE = 64

ZE_RESULT_NAMES = {
    0: "ZE_RESULT_SUCCESS",
    1: "ZE_RESULT_NOT_READY",
    0x70000001: "ZE_RESULT_ERROR_DEVICE_LOST",
    0x70000002: "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY",
    0x70000003: "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    0x70000004: "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE",
    0x70000005: "ZE_RESULT_ERROR_MODULE_LINK_FAILURE",
    0x70000006: "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET",
    0x70000007: "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE",
    0x70010000: "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS",
    0x70010001: "ZE_RESULT_ERROR_NOT_AVAILABLE",
    0x70020000: "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE",
    0x78000001: "ZE_RESULT_ERROR_UNINITIALIZED",
    0x78000002: "ZE_RESULT_ERROR_UNSUPPORTED_VERSION",
    0x78000003: "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE",
    0x78000004: "ZE_RESULT_ERROR_INVALID_ARGUMENT",
    0x78000005: "ZE_RESULT_ERROR_INVALID_NULL_HANDLE",
    0x78000006: "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE",
    0x78000007: "ZE_RESULT_ERROR_INVALID_NULL_POINTER",
    0x78000008: "ZE_RESULT_ERROR_INVALID_SIZE",
    0x78000009: "ZE_RESULT_ERROR_UNSUPPORTED_SIZE",
    0x7800000A: "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT",
    0x7800000B: "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT",
    0x7800000C: "ZE_RESULT_ERROR_INVALID_ENUMERATION",
}


def zr(code):
    code &= 0xFFFFFFFF
    if code >= 0x80000000:
        code -= 0x100000000
    return ZE_RESULT_NAMES.get(code, "0x%08X" % (code & 0xFFFFFFFF))


def ok(code):
    return code == 0


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
class ZeDriverUuid(Structure):
    _fields_ = [("id", c_uint8 * 16)]


class ZeDeviceUuid(Structure):
    _fields_ = [("id", c_uint8 * 16)]


class ZeDriverProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("uuid", ZeDriverUuid),
        ("driverVersion", c_uint32),
    ]


class ZeDriverIpcProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
    ]


class ZeDeviceProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("type", c_uint32),
        ("vendorId", c_uint32),
        ("deviceId", c_uint32),
        ("flags", c_uint32),
        ("subdeviceId", c_uint32),
        ("coreClockRate", c_uint32),
        ("maxMemAllocSize", c_uint64),
        ("maxHardwareContexts", c_uint32),
        ("maxCommandQueuePriority", c_uint32),
        ("numThreadsPerEU", c_uint32),
        ("physicalEUSimdWidth", c_uint32),
        ("numEUsPerSubslice", c_uint32),
        ("numSubslicesPerSlice", c_uint32),
        ("numSlices", c_uint32),
        ("timerResolution", c_uint64),
        ("timestampValidBits", c_uint32),
        ("kernelTimestampValidBits", c_uint32),
        ("uuid", ZeDeviceUuid),
        ("name", c_char * 256),
    ]


class ZeDeviceComputeProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("maxTotalGroupSize", c_uint32),
        ("maxGroupSizeX", c_uint32),
        ("maxGroupSizeY", c_uint32),
        ("maxGroupSizeZ", c_uint32),
        ("maxGroupCountX", c_uint32),
        ("maxGroupCountY", c_uint32),
        ("maxGroupCountZ", c_uint32),
        ("maxSharedLocalMemory", c_uint32),
        ("numSubGroupSizes", c_uint32),
        ("subGroupSizes", c_uint32 * 8),
    ]


class ZeDeviceMemoryProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("maxClockRate", c_uint32),
        ("maxBusWidth", c_uint32),
        ("totalSize", c_uint64),
        ("name", c_char * 256),
    ]


class ZeDeviceP2PProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
    ]


class ZeContextDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
    ]


class ZeDeviceMemAllocDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("ordinal", c_uint32),
    ]


class ZeHostMemAllocDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
    ]


class ZeIpcMemHandle(Structure):
    _fields_ = [("data", c_char * ZE_MAX_IPC_HANDLE_SIZE)]


class ZexIpcCounterBasedEventHandle(Structure):
    _fields_ = [("data", c_char * ZE_MAX_IPC_HANDLE_SIZE)]


class ZePciAddressExt(Structure):
    _fields_ = [
        ("domain", c_uint32),
        ("bus", c_uint32),
        ("device", c_uint32),
        ("function", c_uint32),
    ]


class ZePciSpeedExt(Structure):
    _fields_ = [
        ("genVersion", c_int32),
        ("width", c_int32),
        ("maxBandwidth", c_int64),
    ]


class ZePciExtProperties(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("address", ZePciAddressExt),
        ("maxSpeed", ZePciSpeedExt),
    ]


class ZexCounterBasedEventDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_ext_t),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("signalScope", c_uint32),
        ("waitScope", c_uint32),
    ]


class ZeDriverExtensionProperties(Structure):
    _fields_ = [
        ("name", c_char * 256),
        ("version", c_uint32),
    ]


# ---------------------------------------------------------------------------
# Loader binding
# ---------------------------------------------------------------------------
def load_loader():
    try:
        lib = ctypes.WinDLL("ze_loader.dll")
    except OSError as e:
        print("FATAL: cannot load ze_loader.dll:", e)
        raise SystemExit(2)
    return lib


def bind(lib, name, restype, argtypes):
    fn = getattr(lib, name, None)
    if fn is None:
        return None
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def setup(loader):
    f = {}
    f["zeInit"] = bind(loader, "zeInit", ze_result_t, [c_uint32])
    f["zeDriverGet"] = bind(loader, "zeDriverGet", ze_result_t,
                            [POINTER(c_uint32), POINTER(ze_handle)])
    f["zeDriverGetApiVersion"] = bind(loader, "zeDriverGetApiVersion", ze_result_t,
                                      [ze_handle, POINTER(c_uint32)])
    f["zeDriverGetProperties"] = bind(loader, "zeDriverGetProperties", ze_result_t,
                                      [ze_handle, POINTER(ZeDriverProperties)])
    f["zeDriverGetIpcProperties"] = bind(loader, "zeDriverGetIpcProperties", ze_result_t,
                                         [ze_handle, POINTER(ZeDriverIpcProperties)])
    f["zeDriverGetExtensionProperties"] = bind(
        loader, "zeDriverGetExtensionProperties", ze_result_t,
        [ze_handle, POINTER(c_uint32), POINTER(ZeDriverExtensionProperties)])
    f["zeDriverGetExtensionFunctionAddress"] = bind(
        loader, "zeDriverGetExtensionFunctionAddress", ze_result_t,
        [ze_handle, c_char_p, POINTER(c_void_p)])
    f["zeDeviceGet"] = bind(loader, "zeDeviceGet", ze_result_t,
                            [ze_handle, POINTER(c_uint32), POINTER(ze_handle)])
    f["zeDeviceGetProperties"] = bind(loader, "zeDeviceGetProperties", ze_result_t,
                                      [ze_handle, POINTER(ZeDeviceProperties)])
    f["zeDeviceGetComputeProperties"] = bind(
        loader, "zeDeviceGetComputeProperties", ze_result_t,
        [ze_handle, POINTER(ZeDeviceComputeProperties)])
    f["zeDeviceGetMemoryProperties"] = bind(
        loader, "zeDeviceGetMemoryProperties", ze_result_t,
        [ze_handle, POINTER(c_uint32), POINTER(ZeDeviceMemoryProperties)])
    f["zeDeviceCanAccessPeer"] = bind(
        loader, "zeDeviceCanAccessPeer", ze_result_t,
        [ze_handle, ze_handle, POINTER(ze_bool_t)])
    f["zeDeviceGetP2PProperties"] = bind(
        loader, "zeDeviceGetP2PProperties", ze_result_t,
        [ze_handle, ze_handle, POINTER(ZeDeviceP2PProperties)])
    f["zeDevicePciGetPropertiesExt"] = bind(
        loader, "zeDevicePciGetPropertiesExt", ze_result_t,
        [ze_handle, POINTER(ZePciExtProperties)])
    f["zeContextCreate"] = bind(loader, "zeContextCreate", ze_result_t,
                                [ze_handle, POINTER(ZeContextDesc), POINTER(ze_handle)])
    f["zeContextDestroy"] = bind(loader, "zeContextDestroy", ze_result_t, [ze_handle])
    f["zeMemAllocDevice"] = bind(
        loader, "zeMemAllocDevice", ze_result_t,
        [ze_handle, POINTER(ZeDeviceMemAllocDesc), ctypes.c_size_t,
         ctypes.c_size_t, ze_handle, POINTER(c_void_p)])
    f["zeMemAllocHost"] = bind(
        loader, "zeMemAllocHost", ze_result_t,
        [ze_handle, POINTER(ZeHostMemAllocDesc), ctypes.c_size_t, ctypes.c_size_t,
         POINTER(c_void_p)])
    f["zeMemGetIpcHandle"] = bind(loader, "zeMemGetIpcHandle", ze_result_t,
                                  [ze_handle, c_void_p, POINTER(ZeIpcMemHandle)])
    f["zeMemOpenIpcHandle"] = bind(
        loader, "zeMemOpenIpcHandle", ze_result_t,
        [ze_handle, ze_handle, ZeIpcMemHandle, c_uint32, POINTER(c_void_p)])
    f["zeMemCloseIpcHandle"] = bind(loader, "zeMemCloseIpcHandle", ze_result_t,
                                    [ze_handle, c_void_p])
    f["zeMemFree"] = bind(loader, "zeMemFree", ze_result_t, [ze_handle, c_void_p])
    f["zeEventHostSignal"] = bind(loader, "zeEventHostSignal", ze_result_t, [ze_handle])
    f["zeEventHostSynchronize"] = bind(loader, "zeEventHostSynchronize", ze_result_t,
                                       [ze_handle, c_uint64])
    f["zeEventDestroy"] = bind(loader, "zeEventDestroy", ze_result_t, [ze_handle])
    return f


# zex prototypes resolved either from the loader export table or from the
# driver extension table.
ZEX_CREATE2_PROTO = CFUNCTYPE(ze_result_t, ze_handle, ze_handle,
                              POINTER(ZexCounterBasedEventDesc), POINTER(ze_handle))
ZEX_GET_IPC_PROTO = CFUNCTYPE(ze_result_t, ze_handle,
                              POINTER(ZexIpcCounterBasedEventHandle))
ZEX_OPEN_IPC_PROTO = CFUNCTYPE(ze_result_t, ze_handle,
                               ZexIpcCounterBasedEventHandle, POINTER(ze_handle))
ZEX_CLOSE_IPC_PROTO = CFUNCTYPE(ze_result_t, ze_handle)

ZEX_FUNCS = {
    "zexCounterBasedEventCreate2": ZEX_CREATE2_PROTO,
    "zexCounterBasedEventGetIpcHandle": ZEX_GET_IPC_PROTO,
    "zexCounterBasedEventOpenIpcHandle": ZEX_OPEN_IPC_PROTO,
    "zexCounterBasedEventCloseIpcHandle": ZEX_CLOSE_IPC_PROTO,
}


def loader_export(lib, name):
    try:
        return ctypes.cast(getattr(lib, name), c_void_p).value
    except AttributeError:
        return None


def ext_function(driver, name):
    addr = c_void_p()
    try:
        rc = F["zeDriverGetExtensionFunctionAddress"](driver, name.encode(), byref(addr))
    except Exception:
        return None, "loader call failed"
    if rc != 0:
        return None, zr(rc)
    if not addr.value:
        return None, "null address"
    return addr.value, None


def decode_loader_version(path=r"C:\Windows\System32\ze_loader.dll"):
    # GetFileVersionInfo via version.dll
    try:
        vd = ctypes.WinDLL("version")
        size = vd.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not vd.GetFileVersionInfoW(path, 0, size, buf):
            return None
        val = c_void_p()
        length = c_uint32()
        if not vd.VerQueryValueW(buf, "\\", byref(val), byref(length)):
            return None
        # VS_FIXEDFILEINFO
        class VS_FIXEDFILEINFO(Structure):
            _fields_ = [
                ("dwSignature", c_uint32), ("dwStrucVersion", c_uint32),
                ("dwFileVersionMS", c_uint32), ("dwFileVersionLS", c_uint32),
                ("dwProductVersionMS", c_uint32), ("dwProductVersionLS", c_uint32),
                ("dwFileFlagsMask", c_uint32), ("dwFileFlags", c_uint32),
                ("dwFileOS", c_uint32), ("dwFileType", c_uint32),
                ("dwFileSubtype", c_uint32), ("dwFileDateMS", c_uint32),
                ("dwFileDateLS", c_uint32),
            ]
        info = cast(val, POINTER(VS_FIXEDFILEINFO)).contents
        ms, ls = info.dwFileVersionMS, info.dwFileVersionLS
        return "%d.%d.%d.%d" % (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except Exception as e:
        return "unknown (%s)" % e


F = {}


def main():
    global F
    print("=" * 78)
    print("S02 L0 capability spike (read-only, no perf measurements)")
    print("=" * 78)

    loader = load_loader()
    F = setup(loader)

    missing = [k for k, v in F.items() if v is None]
    print("\n[0] Loader bind check")
    print("    ze_loader.dll file version :", decode_loader_version())
    print("    core symbols missing       :", missing if missing else "none")

    # -- step 1: init + drivers --------------------------------------------
    print("\n[1] zeInit + drivers")
    rc = F["zeInit"](ZE_INIT_FLAG_GPU_ONLY)
    print("    zeInit(ZE_INIT_FLAG_GPU_ONLY) ->", zr(rc))
    if not ok(rc):
        return

    ndrv = c_uint32(0)
    rc = F["zeDriverGet"](byref(ndrv), None)
    print("    zeDriverGet(count) ->", zr(rc), "count =", ndrv.value)
    if not ok(rc) or ndrv.value == 0:
        print("    no L0 drivers found")
        return
    drivers = (ze_handle * ndrv.value)()
    rc = F["zeDriverGet"](byref(ndrv), drivers)
    print("    zeDriverGet(list)  ->", zr(rc))

    all_devices = []  # (driver, handle, props)

    for di in range(ndrv.value):
        drv = drivers[di]
        ver = c_uint32(0)
        rc = F["zeDriverGetApiVersion"](drv, byref(ver))
        v = ver.value
        print("    driver[%d] apiVersion -> %s  (%d.%d)" %
              (di, zr(rc), v >> 16, v & 0xFFFF))
        dp = ZeDriverProperties(stype=ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES)
        rc = F["zeDriverGetProperties"](drv, byref(dp))
        print("    driver[%d] driverVersion=0x%08X  (%s)" %
              (di, dp.driverVersion, zr(rc)))
        ipc = ZeDriverIpcProperties(stype=ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES)
        rc = F["zeDriverGetIpcProperties"](drv, byref(ipc))
        flags = []
        if ipc.flags & ZE_IPC_PROPERTY_FLAG_MEMORY:
            flags.append("MEMORY")
        if ipc.flags & ZE_IPC_PROPERTY_FLAG_EVENT_POOL:
            flags.append("EVENT_POOL")
        print("    driver[%d] ipcProperties flags=0x%X [%s]  (%s)" %
              (di, ipc.flags, ",".join(flags) or "-", zr(rc)))

        # extension list
        nextn = c_uint32(0)
        rc = F["zeDriverGetExtensionProperties"](drv, byref(nextn), None)
        exts = []
        if ok(rc) and nextn.value:
            arr = (ZeDriverExtensionProperties * nextn.value)()
            rc = F["zeDriverGetExtensionProperties"](drv, byref(nextn), arr)
            for e in arr:
                exts.append(e.name.decode("ascii", "replace"))
        print("    driver[%d] extensions: %d" % (di, len(exts)))
        interesting = [e for e in exts if "counter_based" in e.lower()
                       or "ipc" in e.lower() or "p2p" in e.lower()]
        print("    driver[%d] ipc/p2p/ext-of-interest: %s" % (di, interesting or "-"))
        has_cbe = "ZEX_counter_based_event" in exts

        # -- step 2: devices -----------------------------------------------
        ndev = c_uint32(0)
        rc = F["zeDeviceGet"](drv, byref(ndev), None)
        print("    driver[%d] zeDeviceGet count -> %s, n=%d" % (di, zr(rc), ndev.value))
        if not ok(rc) or ndev.value == 0:
            continue
        devs = (ze_handle * ndev.value)()
        rc = F["zeDeviceGet"](drv, byref(ndev), devs)
        for k in range(ndev.value):
            h = devs[k]
            p = ZeDeviceProperties(stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES)
            rc = F["zeDeviceGetProperties"](h, byref(p))
            name = p.name.decode("ascii", "replace")
            print("      device[%d.%d] '%s' type=%d vid=0x%04X did=0x%04X "
                  "numSlices=%d numSubslicesPerSlice=%d numEUsPerSubslice=%d "
                  "coreClock=%uMHz" %
                  (di, k, name, p.type, p.vendorId, p.deviceId, p.numSlices,
                   p.numSubslicesPerSlice, p.numEUsPerSubslice, p.coreClockRate))
            cp = ZeDeviceComputeProperties(stype=ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES)
            rc = F["zeDeviceGetComputeProperties"](h, byref(cp))
            print("      device[%d.%d] compute: maxGroup=%u subGroups=%s" %
                  (di, k, cp.maxTotalGroupSize,
                   [cp.subGroupSizes[i] for i in range(cp.numSubGroupSizes)]))
            nmem = c_uint32(0)
            F["zeDeviceGetMemoryProperties"](h, byref(nmem), None)
            memsz = None
            if nmem.value:
                memarr = (ZeDeviceMemoryProperties * nmem.value)()
                for m in memarr:
                    m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
                rc = F["zeDeviceGetMemoryProperties"](h, byref(nmem), memarr)
                sizes = ["%.2f GiB" % (m.totalSize / (1 << 30)) for m in memarr]
                memsz = sizes[0]
                print("      device[%d.%d] mem: n=%d sizes=%s" %
                      (di, k, nmem.value, sizes))
            pci = ZePciExtProperties(stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES)
            rc = F["zeDevicePciGetPropertiesExt"](h, byref(pci))
            pci_str = "-"
            if ok(rc):
                a = pci.address
                pci_str = "%04X:%02X:%02X.%X gen%d x%d (%.1f GB/s)" % (
                    a.domain, a.bus, a.device, a.function,
                    pci.maxSpeed.genVersion, pci.maxSpeed.width,
                    pci.maxSpeed.maxBandwidth / 1e9 if pci.maxSpeed.maxBandwidth > 0 else 0)
            print("      device[%d.%d] pci: %s  (%s)" % (di, k, pci_str, zr(rc)))

            all_devices.append({
                "driver": drv, "handle": h, "props": p,
                "name": name, "mem": memsz, "has_cbe": has_cbe,
                "index": len(all_devices),
            })

    n = len(all_devices)
    print("\n    TOTAL GPU devices (ZE_DEVICE_TYPE_GPU in enumeration): %d" % n)
    if n == 0:
        return

    # -- step 3: contexts --------------------------------------------------
    print("\n[3] zeContextCreate (one per device, no flags)")
    ctxs = []
    for dev in all_devices:
        c = ze_handle()
        desc = ZeContextDesc(stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, flags=0)
        rc = F["zeContextCreate"](dev["driver"], byref(desc), byref(c))
        dev["ctx"] = c
        dev["ctx_rc"] = rc
        print("    ctx[%d] '%s' -> %s" % (dev["index"], dev["name"], zr(rc)))

    # -- step 4: peer matrix -----------------------------------------------
    print("\n[4] zeDeviceCanAccessPeer matrix (rows = accessor i, cols = peer j)")
    header = "        " + "".join("j=%-12d" % j for j in range(n))
    print(header)
    can = [[False] * n for _ in range(n)]
    for i in range(n):
        row = "    i=%-3d" % i
        for j in range(n):
            val = ze_bool_t(0)
            rc = F["zeDeviceCanAccessPeer"](all_devices[i]["handle"],
                                            all_devices[j]["handle"], byref(val))
            can[i][j] = bool(val.value) and ok(rc)
            cell = "Y" if can[i][j] else ("n" if ok(rc) else "E")
            row += "%-13s" % ("%s(%s)" % (cell, zr(rc).replace("ZE_RESULT_", "")))
        print(row)

    print("\n[4b] zeDeviceGetP2PProperties for all ordered pairs")
    p2p = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            pp = ZeDeviceP2PProperties(stype=ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES)
            rc = F["zeDeviceGetP2PProperties"](all_devices[i]["handle"],
                                               all_devices[j]["handle"], byref(pp))
            p2p[i][j] = (pp.flags, rc)
            fl = []
            if pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS:
                fl.append("ACCESS")
            if pp.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS:
                fl.append("ATOMICS")
            print("    %d->%d : flags=0x%X [%s]  rc=%s" %
                  (i, j, pp.flags, ",".join(fl) or "-", zr(rc)))

    # -- step 5: IPC memory ------------------------------------------------
    print("\n[5] IPC device memory (1 MiB, align 64)")
    SIZE = 1 << 20

    # extension prototypes (ZEX_mem_ipc_handles; documented Linux-only)
    ZEXMEMGET_PROTO = CFUNCTYPE(ze_result_t, ze_handle, c_void_p,
                                POINTER(c_uint32), POINTER(ZeIpcMemHandle))
    ZEXMEMOPEN_PROTO = CFUNCTYPE(ze_result_t, ze_handle, ze_handle, c_uint32,
                                 POINTER(ZeIpcMemHandle), c_uint32, POINTER(c_void_p))
    gaddr, gerr = ext_function(all_devices[0]["driver"], "zexMemGetIpcHandles")
    oaddr, oerr = ext_function(all_devices[0]["driver"], "zexMemOpenIpcHandles")
    zex_mem_get = ZEXMEMGET_PROTO(gaddr) if gaddr else None
    zex_mem_open = ZEXMEMOPEN_PROTO(oaddr) if oaddr else None
    print("    zexMemGetIpcHandles  resolved: %s" % ("yes" if zex_mem_get else "no (%s)" % gerr))
    print("    zexMemOpenIpcHandles resolved: %s" % ("yes" if zex_mem_open else "no (%s)" % oerr))

    for i in range(n):
        dev = all_devices[i]
        ctx = dev["ctx"]
        if not ok(dev["ctx_rc"]):
            print("    dev[%d] skip (no context)" % i)
            continue
        ptr = c_void_p()
        adesc = ZeDeviceMemAllocDesc(stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                     flags=0, ordinal=0)
        rc = F["zeMemAllocDevice"](ctx, byref(adesc), SIZE, 64,
                                   dev["handle"], byref(ptr))
        print("    dev[%d] zeMemAllocDevice -> %s ptr=%s" %
              (i, zr(rc), hex(ptr.value) if ptr.value else "None"))
        if not ok(rc) or not ptr.value:
            continue
        dev["ptr"] = ptr

        # (a) standard zeMemGetIpcHandle
        ipc = ZeIpcMemHandle()
        rc_std = F["zeMemGetIpcHandle"](ctx, ptr, byref(ipc))
        print("    dev[%d] zeMemGetIpcHandle (core) -> %s" % (i, zr(rc_std)))

        # (a2) standard on host USM alloc
        hptr = c_void_p()
        hdesc = ZeHostMemAllocDesc(stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, flags=0)
        rc_h = F["zeMemAllocHost"](ctx, byref(hdesc), SIZE, 64, byref(hptr))
        ipch = ZeIpcMemHandle()
        rc_std_h = F["zeMemGetIpcHandle"](ctx, hptr, byref(ipch)) if ok(rc_h) else -1
        print("    dev[%d] zeMemAllocHost -> %s ; zeMemGetIpcHandle(host) -> %s" %
              (i, zr(rc_h), zr(rc_std_h) if rc_std_h != -1 else "skipped"))

        # (b) ZEX_mem_ipc_handles extension
        if zex_mem_get:
            cnt = c_uint32(0)
            rcc = zex_mem_get(ctx, ptr, byref(cnt), None)   # query count
            arr = (ZeIpcMemHandle * max(1, cnt.value))()
            rc2 = zex_mem_get(ctx, ptr, byref(cnt), arr) if ok(rcc) else rcc
            print("    dev[%d] zexMemGetIpcHandles: query=%s n=%d get=%s" %
                  (i, zr(rcc), cnt.value, zr(rc2)))
            if ok(rc2) and cnt.value and zex_mem_open:
                for j in range(n):
                    pj = c_void_p()
                    rco = zex_mem_open(all_devices[j]["ctx"], all_devices[j]["handle"],
                                       cnt.value, arr, 0, byref(pj))
                    print("    dev[%d] zexMemOpenIpcHandles(->%d) -> %s ptr=%s" %
                          (i, j, zr(rco), hex(pj.value) if pj.value else "None"))
                    if ok(rco) and pj.value:
                        F["zeMemCloseIpcHandle"](all_devices[j]["ctx"], pj)

        if ok(rc_std):
            # sanity: open on own device then each peer
            p2 = c_void_p()
            rc = F["zeMemOpenIpcHandle"](ctx, dev["handle"], ipc, 0, byref(p2))
            print("    dev[%d] zeMemOpenIpcHandle(self) -> %s" % (i, zr(rc)))
            if ok(rc) and p2.value:
                F["zeMemCloseIpcHandle"](ctx, p2)
            for j in range(n):
                if j == i:
                    continue
                pj = c_void_p()
                rc = F["zeMemOpenIpcHandle"](all_devices[j]["ctx"],
                                             all_devices[j]["handle"], ipc, 0, byref(pj))
                print("    dev[%d] zeMemOpenIpcHandle(->%d) -> %s" %
                      (i, j, zr(rc)))
                if ok(rc) and pj.value:
                    F["zeMemCloseIpcHandle"](all_devices[j]["ctx"], pj)

    # -- step 6: IPC counter-based events ----------------------------------
    print("\n[6] counter-based events (zex): symbol availability + IPC variants")
    resolved = {}
    for fname, proto in ZEX_FUNCS.items():
        direct = loader_export(loader, fname)
        if direct:
            resolved[fname] = ("loader_export", proto(direct))
            print("    %s : exported by ze_loader.dll" % fname)
        else:
            addr, err = ext_function(all_devices[0]["driver"], fname)
            if addr:
                resolved[fname] = ("driver_extension", proto(addr))
                print("    %s : via zeDriverGetExtensionFunctionAddress" % fname)
            else:
                print("    %s : NOT AVAILABLE (loader export absent; ext: %s)" %
                      (fname, err))

    create2 = resolved.get("zexCounterBasedEventCreate2", (None, None))[1]
    get_ipc = resolved.get("zexCounterBasedEventGetIpcHandle", (None, None))[1]
    open_ipc = resolved.get("zexCounterBasedEventOpenIpcHandle", (None, None))[1]
    close_ipc = resolved.get("zexCounterBasedEventCloseIpcHandle", (None, None))[1]

    def make_event(dev, flags, tag):
        desc = ZexCounterBasedEventDesc(
            stype=ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC, pNext=None, flags=flags,
            signalScope=ZE_EVENT_SCOPE_FLAG_HOST, waitScope=ZE_EVENT_SCOPE_FLAG_HOST)
        ev = ze_handle()
        rc = create2(dev["ctx"], dev["handle"], byref(desc), byref(ev))
        print("      create2 flags=0x%X (%s) -> %s ev=%s" %
              (flags, tag, zr(rc), bool(ev.value)))
        return ev if ok(rc) and ev.value else None

    if not create2:
        print("    -> cannot create events: zexCounterBasedEventCreate2 unavailable")
    elif not ok(all_devices[0]["ctx_rc"]):
        print("    -> no context")
    else:
        dev0 = all_devices[0]
        # (i) local (non-IPC) counter event: verify same-device signal/wait works
        ev_local = make_event(dev0, ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE |
                              ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE, "local")
        if ev_local:
            q0 = F["zeEventHostSynchronize"](ev_local, 0)          # query before signal
            s = F["zeEventHostSignal"](ev_local)                    # host signal
            q1 = F["zeEventHostSynchronize"](ev_local, 1_000_000_000)  # wait after signal
            print("      local smoke: preQuery(0ns)=%s signal=%s postQuery=%s" %
                  (zr(q0), zr(s), zr(q1)))
            F["zeEventDestroy"](ev_local)

        # (ii) IPC variants
        variants = [
            (ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE |
             ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE |
             ZEX_COUNTER_BASED_EVENT_FLAG_IPC, "IMMEDIATE|HOST_VISIBLE|IPC"),
            (ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE |
             ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE |
             ZEX_COUNTER_BASED_EVENT_FLAG_IPC, "NON_IMMEDIATE|HOST_VISIBLE|IPC"),
            (ZEX_COUNTER_BASED_EVENT_FLAG_IPC, "IPC only"),
        ]
        for flags, tag in variants:
            ev = make_event(dev0, flags, tag)
            if not ev:
                continue
            if get_ipc:
                ipch = ZexIpcCounterBasedEventHandle()
                rc = get_ipc(ev, byref(ipch))
                print("      GetIpcHandle -> %s" % zr(rc))
                if ok(rc) and open_ipc:
                    for j in range(1, n):
                        evj = ze_handle()
                        rco = open_ipc(all_devices[j]["ctx"], ipch, byref(evj))
                        print("      OpenIpcHandle(->%d) -> %s" % (j, zr(rco)))
                        if ok(rco) and evj.value:
                            s = F["zeEventHostSignal"](ev)
                            w = F["zeEventHostSynchronize"](evj, 1_000_000_000)
                            print("      cross smoke signal(ev0)=%s wait(ev%d)=%s" %
                                  (zr(s), j, zr(w)))
                            if close_ipc:
                                close_ipc(evj)
            F["zeEventDestroy"](ev)

    # -- cleanup -----------------------------------------------------------
    print("\n[7] cleanup")
    for dev in all_devices:
        if "ptr" in dev and dev["ptr"].value:
            rc = F["zeMemFree"](dev["ctx"], dev["ptr"])
            print("    zeMemFree dev[%d] -> %s" % (dev["index"], zr(rc)))
        if ok(dev.get("ctx_rc", 1)) and dev.get("ctx") and dev["ctx"].value:
            rc = F["zeContextDestroy"](dev["ctx"])
            print("    zeContextDestroy dev[%d] -> %s" % (dev["index"], zr(rc)))

    print("\ndone.")


if __name__ == "__main__":
    main()
