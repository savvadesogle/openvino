#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# S03 L0 microbenchmark spike (read-only).
#
# Measures inter-device data-transfer paths on 2+ discrete Intel GPUs on Windows
# (Level Zero over ze_loader.dll), to decide Gate G0 (default TP collective path).
#
# S02 established that on this machine (2x Arc A770, Windows) there is NO
# cross-device peer access (zeDeviceCanAccessPeer=false) and NO IPC memory/event
# export (UNSUPPORTED_FEATURE / OUT_OF_DEVICE_MEMORY). Therefore the only
# candidate paths are measured here explicitly:
#
#   (C) same-device device->device copy   (local bandwidth baseline)
#   (A) host-USM staging: device_i -> pinned host USM -> device_j
#       (device->host and host->device measured separately + end-to-end)
#   (B) shared USM (zeMemAllocShared): write by device i, read by device j
#   (D) any cross-device route that actually works is flagged with its route
#   peer-copy i->j direct : recorded as NA (see S02 / l0-spike.md)
#
# Copy engine: regular command list + command queue, host sync via
# zeCommandQueueSynchronize (UINT64_MAX). Classic zeEvent* is deliberately NOT
# used: in S02 zeEventCreate from ctypes crashed the driver (see STATE.md).
#
# Reuses types/loader helpers from l0_capability.py (same directory).
#
# Usage:  python l0_bench.py
#
# Read-only wrt OpenVINO. Allocates/frees USM only; frees everything on exit.

import ctypes
import os
import statistics
import sys
import time
from ctypes import (POINTER, Structure, byref, c_char, c_int32, c_size_t,
                    c_uint8, c_uint32, c_uint64, c_void_p)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l0_capability as l0  # noqa: E402

ze_result_t = l0.ze_result_t
ze_handle = l0.ze_handle
ze_structure_type_t = l0.ze_structure_type_t
zr = l0.zr
ok = l0.ok
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = l0.ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = l0.ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
ZE_STRUCTURE_TYPE_CONTEXT_DESC = l0.ZE_STRUCTURE_TYPE_CONTEXT_DESC

# From ze_api.h 1.33.1 (system header)
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x0E
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0x0F
ZE_COMMAND_QUEUE_FLAG_IN_ORDER = (1 << 0)
ZE_COMMAND_LIST_FLAG_IN_ORDER = (1 << 0)

UINT64_MAX = 0xFFFFFFFFFFFFFFFF
UINT32_MAX = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Structures (from ze_api.h 1.33.1)
# ---------------------------------------------------------------------------
class ZeCommandQueueDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("ordinal", c_uint32),
        ("index", c_uint32),
        ("flags", c_uint32),
        ("mode", c_int32),
        ("priority", c_int32),
    ]


class ZeCommandListDesc(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),
        ("pNext", c_void_p),
        ("commandQueueGroupOrdinal", c_uint32),
        ("flags", c_uint32),
    ]


# ---------------------------------------------------------------------------
# Extra loader bindings (command queues/lists + shared USM)
# ---------------------------------------------------------------------------
def setup_bench(F):
    def b(name, restype, argtypes):
        fn = getattr(l0_loader, name, None)
        if fn is None:
            raise SystemExit("FATAL: ze_loader.dll does not export %s" % name)
        fn.restype = restype
        fn.argtypes = argtypes
        F[name] = fn

    b("zeCommandQueueCreate", ze_result_t,
      [ze_handle, ze_handle, POINTER(ZeCommandQueueDesc), POINTER(ze_handle)])
    b("zeCommandQueueDestroy", ze_result_t, [ze_handle])
    b("zeCommandQueueExecuteCommandLists", ze_result_t,
      [ze_handle, c_uint32, POINTER(ze_handle), ze_handle])
    b("zeCommandQueueSynchronize", ze_result_t, [ze_handle, c_uint64])
    b("zeCommandListCreate", ze_result_t,
      [ze_handle, ze_handle, POINTER(ZeCommandListDesc), POINTER(ze_handle)])
    b("zeCommandListDestroy", ze_result_t, [ze_handle])
    b("zeCommandListClose", ze_result_t, [ze_handle])
    b("zeCommandListReset", ze_result_t, [ze_handle])
    b("zeCommandListAppendMemoryCopy", ze_result_t,
      [ze_handle, c_void_p, c_void_p, c_size_t, ze_handle, c_uint32,
       POINTER(ze_handle)])
    b("zeCommandListAppendBarrier", ze_result_t,
      [ze_handle, ze_handle, c_uint32, POINTER(ze_handle)])
    b("zeMemAllocShared", ze_result_t,
      [ze_handle, POINTER(l0.ZeDeviceMemAllocDesc),
       POINTER(l0.ZeHostMemAllocDesc), c_size_t, c_size_t, ze_handle,
       POINTER(c_void_p)])


# ---------------------------------------------------------------------------
# Per-device engine (context + command queue + reusable command list)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, dev):
        self.dev = dev
        self.ctx = dev["ctx"]
        self.hdev = dev["handle"]
        self.queue = ze_handle()
        self.list = ze_handle()

    def create(self):
        qdesc = ZeCommandQueueDesc(stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   pNext=None, ordinal=0, index=0,
                                   flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                                   mode=0, priority=0)
        rc = F["zeCommandQueueCreate"](self.ctx, self.hdev, byref(qdesc),
                                       byref(self.queue))
        if not ok(rc):
            return "zeCommandQueueCreate: %s" % zr(rc)
        ldesc = ZeCommandListDesc(stype=ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                  pNext=None, commandQueueGroupOrdinal=0,
                                  flags=ZE_COMMAND_LIST_FLAG_IN_ORDER)
        rc = F["zeCommandListCreate"](self.ctx, self.hdev, byref(ldesc),
                                      byref(self.list))
        if not ok(rc):
            return "zeCommandListCreate: %s" % zr(rc)
        return None

    def run_batch(self, dst, src, size, batch):
        F["zeCommandListReset"](self.list)
        for _ in range(batch):
            rc = F["zeCommandListAppendMemoryCopy"](
                self.list, dst, src, size, None, 0, None)
            if not ok(rc):
                return None, "appendMemoryCopy: %s" % zr(rc)
        rc = F["zeCommandListClose"](self.list)
        if not ok(rc):
            return None, "listClose: %s" % zr(rc)
        arr = (ze_handle * 1)(self.list)
        t0 = time.perf_counter_ns()
        rc = F["zeCommandQueueExecuteCommandLists"](self.queue, 1, arr, None)
        if not ok(rc):
            return None, "queueExecute: %s" % zr(rc)
        rc = F["zeCommandQueueSynchronize"](self.queue, UINT64_MAX)
        t1 = time.perf_counter_ns()
        if not ok(rc):
            return None, "queueSynchronize: %s" % zr(rc)
        return (t1 - t0), None

    def destroy(self):
        if self.list.value:
            F["zeCommandListDestroy"](self.list)
        if self.queue.value:
            F["zeCommandQueueDestroy"](self.queue)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def stats(times_ns):
    return {
        "min": min(times_ns),
        "p10": statistics.quantiles(times_ns, n=10)[0] if len(times_ns) >= 10 else min(times_ns),
        "median": statistics.median(times_ns),
        "p90": statistics.quantiles(times_ns, n=10)[-1] if len(times_ns) >= 10 else max(times_ns),
        "max": max(times_ns),
    }


def measure(engine, dst, src, size, batch, reps, warmup):
    """Returns (per-op median ns, latency stats ns dict, error)."""
    err = None
    for _ in range(warmup):
        _, err = engine.run_batch(dst, src, size, batch)
        if err:
            return None, None, err
    # throughput: batch>1 amortises submission overhead
    thru = []
    for _ in range(reps):
        ns, err = engine.run_batch(dst, src, size, batch)
        if err:
            return None, None, err
        thru.append(ns / batch)
    # latency: single op, host-observed (includes submission + sync)
    lat = []
    for _ in range(reps):
        ns, err = engine.run_batch(dst, src, size, 1)
        if err:
            return None, None, err
        lat.append(ns)
    return statistics.median(thru), stats(lat), None


def to_gbps(size, ns):
    return (size / (ns / 1e9)) / 1e9


def to_us(ns):
    return ns / 1e3


def sizestr(n):
    if n >= 1 << 20:
        return "%d MB" % (n // (1 << 20))
    if n >= 1 << 10:
        return "%d KB" % (n // (1 << 10))
    return "%d B" % n


# ---------------------------------------------------------------------------
# Benchmark body
# ---------------------------------------------------------------------------
def bench_size(size):
    """Returns dict of path -> printable result for both directions."""
    # batch for throughput: keep total enqueued work ~ 32 MB per sync
    batch = max(1, min(128, (32 << 20) // size))
    reps = 20 if size <= (1 << 20) else 8
    warmup = 3
    res = {}

    # (C) same-device baseline: dev->dev within one GPU
    for i in (0, 1):
        eng = engines[i]
        d = devbuf[i]
        med, lat, err = measure(eng, d["b"], d["a"], size, batch, reps, warmup)
        res["same-dev %d->%d" % (i, i)] = (size, med, lat, err, "device USM->device USM")
        del lat

    # (A) staging: device i -> pinned host ; host -> device j
    for i in (0, 1):
        med, lat, err = measure(engines[i], host_ptr, devbuf[i]["a"], size,
                                batch, reps, warmup)
        res["d2h %d->host" % i] = (size, med, lat, err, "device USM->host USM (pinned)")

    for j in (0, 1):
        med, lat, err = measure(engines[j], devbuf[j]["b"], host_ptr, size,
                                batch, reps, warmup)
        res["h2d host->%d" % j] = (size, med, lat, err, "host USM (pinned)->device USM")

    # (A e2e) device i -> host -> device j (sequential: sync between stages)
    for (i, j) in ((0, 1), (1, 0)):
        # warmup
        err = None
        for _ in range(warmup):
            _, e1 = engines[i].run_batch(host_ptr, devbuf[i]["a"], size, batch)
            _, e2 = engines[j].run_batch(devbuf[j]["b"], host_ptr, size, batch)
            err = e1 or e2
            if err:
                break
        if not err:
            times = []
            for _ in range(reps):
                t0 = time.perf_counter_ns()
                _, e1 = engines[i].run_batch(host_ptr, devbuf[i]["a"], size, batch)
                _, e2 = engines[j].run_batch(devbuf[j]["b"], host_ptr, size, batch)
                t1 = time.perf_counter_ns()
                if e1 or e2:
                    err = e1 or e2
                    break
                times.append((t1 - t0) / batch)
            if not err:
                res["stage %d->host->%d" % (i, j)] = (
                    size, statistics.median(times), None, None,
                    "device %d -> host USM -> device %d (2 stages, host sync between)" % (i, j))

    # (B) shared USM: device i writes shared, device j reads shared
    if shared_ok:
        for (i, j) in ((0, 1), (1, 0)):
            # write: dev_i -> shared (queue i)
            med, lat, err = measure(engines[i], shared_ptr, devbuf[i]["a"], size,
                                    batch, reps, warmup)
            res["shared-wr %d->sh" % i] = (size, med, lat, err, "device USM->shared USM")
            # read: shared -> dev_j (queue j) -- cross-context if j != owner
            med, lat, err = measure(engines[j], devbuf[j]["b"], shared_ptr, size,
                                    batch, reps, warmup)
            res["shared-rd sh->%d" % j] = (size, med, lat, err,
                                           "shared USM->device USM (read by device %d)" % j)
    return res


def main():
    global l0_loader, F, engines, devbuf, host_ptr, shared_ptr, shared_ok
    print("=" * 90)
    print("S03 L0 microbench: inter-device transfer paths on", time.strftime("%Y-%m-%d"))
    print("=" * 90)

    l0_loader = l0.load_loader()
    F = l0.setup(l0_loader)
    setup_bench(F)

    rc = F["zeInit"](l0.ZE_INIT_FLAG_GPU_ONLY)
    print("zeInit ->", zr(rc))
    if not ok(rc):
        return

    ndrv = c_uint32(0)
    F["zeDriverGet"](byref(ndrv), None)
    drivers = (ze_handle * ndrv.value)()
    F["zeDriverGet"](byref(ndrv), drivers)

    devs = []
    for di in range(ndrv.value):
        ndev = c_uint32(0)
        if not ok(F["zeDeviceGet"](drivers[di], byref(ndev), None)) or ndev.value == 0:
            continue
        arr = (ze_handle * ndev.value)()
        F["zeDeviceGet"](drivers[di], byref(ndev), arr)
        for k in range(ndev.value):
            p = l0.ZeDeviceProperties(stype=l0.ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES)
            if not ok(F["zeDeviceGetProperties"](arr[k], byref(p))):
                continue
            if p.type != l0.ZE_DEVICE_TYPE_GPU:
                continue
            c = ze_handle()
            cd = l0.ZeContextDesc(stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, flags=0)
            rcc = F["zeContextCreate"](drivers[di], byref(cd), byref(c))
            devs.append({"driver": drivers[di], "handle": arr[k], "name":
                         p.name.decode("ascii", "replace"), "ctx": c, "ctx_rc": rcc})
    print("GPU devices: %d" % len(devs))
    for i, d in enumerate(devs):
        print("  dev[%d] %s ctx=%s" % (i, d["name"], zr(d["ctx_rc"])))
    if len(devs) < 2:
        print("need >= 2 GPUs"); return

    engines = []
    for d in devs:
        e = Engine(d)
        err = e.create()
        engines.append(e)
        print("  engine[%d] queue/list create: %s" % (len(engines) - 1, err or "ok"))
        if err:
            return

    sizes = [(1 << 12), (1 << 16), (1 << 20), (1 << 24), (1 << 27)]  # 4K..128M
    maxsz = sizes[-1]

    # per-device scratch buffers (a = source, b = destination)
    devbuf = []
    for i, d in enumerate(devs):
        entry = {}
        for tag in ("a", "b"):
            p = c_void_p()
            ad = l0.ZeDeviceMemAllocDesc(
                stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, pNext=None,
                flags=0, ordinal=0)
            rca = F["zeMemAllocDevice"](d["ctx"], byref(ad), maxsz, 64,
                                        d["handle"], byref(p))
            entry[tag] = p
            if not ok(rca) or not p.value:
                print("FATAL: device alloc dev%d:%s -> %s" % (i, tag, zr(rca)))
                return
        devbuf.append(entry)

    # pinned host USM (allocated in ctx0)
    host_ptr = c_void_p()
    hd = l0.ZeHostMemAllocDesc(stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                               pNext=None, flags=0)
    rch = F["zeMemAllocHost"](devs[0]["ctx"], byref(hd), maxsz, 64, byref(host_ptr))
    print("pinned host USM alloc (%d MB): %s ptr=%s" %
          (maxsz >> 20, zr(rch), hex(host_ptr.value) if host_ptr.value else "None"))
    if not ok(rch):
        return

    # shared USM (allocated in ctx0, associated with dev0)
    shared_ptr = c_void_p()
    sd = l0.ZeDeviceMemAllocDesc(stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                 pNext=None, flags=0, ordinal=0)
    shd = l0.ZeHostMemAllocDesc(stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                pNext=None, flags=0)
    rcs = F["zeMemAllocShared"](devs[0]["ctx"], byref(sd), byref(shd), maxsz, 64,
                                devs[0]["handle"], byref(shared_ptr))
    shared_ok = ok(rcs) and bool(shared_ptr.value)
    print("shared USM alloc (%d MB): %s ptr=%s" %
          (maxsz >> 20, zr(rcs), hex(shared_ptr.value) if shared_ptr.value else "None"))

    # --- correctness smoke: host->dev->host roundtrip on each device ---------
    print("\n[correctness smoke] host pattern -> dev -> host roundtrip (4 KB)")
    smoke = ctypes.string_at  # placeholder to avoid unused import warnings
    for i, d in enumerate(devs):
        pat = bytes((j * 7 + i) & 0xFF for j in range(1 << 12))
        ctypes.memmove(host_ptr, pat, len(pat))
        _, e1 = engines[i].run_batch(devbuf[i]["a"], host_ptr, len(pat), 1)
        # read back to a fresh host region: reuse host_ptr after copying to a
        # second host allocation
        hb = c_void_p()
        hd2 = l0.ZeHostMemAllocDesc(stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                    pNext=None, flags=0)
        F["zeMemAllocHost"](d["ctx"], byref(hd2), len(pat), 64, byref(hb))
        _, e2 = engines[i].run_batch(hb, devbuf[i]["a"], len(pat), 1)
        got = ctypes.string_at(hb, len(pat))
        verdict = "MATCH" if got == pat else "MISMATCH"
        print("  dev[%d]: h2d=%s d2h=%s -> %s" %
              (i, e1 or "ok", e2 or "ok", verdict if not (e1 or e2) else "ERR"))
        F["zeMemFree"](d["ctx"], hb)

    # --- main sweep ----------------------------------------------------------
    all_res = {}
    for size in sizes:
        print("\n" + "-" * 90)
        print("size = %s  (batch=%d copies per sync; host-observed)" %
              (sizestr(size), max(1, min(128, (32 << 20) // size))))
        print("-" * 90)
        r = bench_size(size)
        all_res[size] = r
        for path, (sz, med, lat, err, route) in r.items():
            if err:
                print("  %-22s : ERROR %s" % (path, err))
                continue
            bw = to_gbps(size, med)
            lat_us = to_us(lat["median"]) if lat else float("nan")
            print("  %-22s : %8.2f GB/s (per-op %7.2f us @median) | "
                  "latency single-op med=%7.2f us p10=%7.2f p90=%7.2f | %s" %
                  (path, bw, to_us(med), lat_us,
                   to_us(lat["p10"]) if lat else float("nan"),
                   to_us(lat["p90"]) if lat else float("nan"), route))

    # --- compact bandwidth matrix -------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY: effective bandwidth GB/s (median of batch per-op)")
    print("=" * 90)
    paths = []
    for size in sizes:
        for p in all_res[size]:
            if p not in paths:
                paths.append(p)
    hdr = "  %-22s" % "path" + "".join("%10s" % sizestr(s) for s in sizes)
    print(hdr)
    for p in paths:
        row = "  %-22s" % p
        for s in sizes:
            v = all_res[s].get(p)
            if v is None or v[3]:
                row += "%10s" % "NA"
            else:
                row += "%10.2f" % to_gbps(s, v[1])
        print(row)

    # --- cleanup ------------------------------------------------------------
    print("\ncleanup")
    if host_ptr.value:
        F["zeMemFree"](devs[0]["ctx"], host_ptr)
    if shared_ok:
        F["zeMemFree"](devs[0]["ctx"], shared_ptr)
    for i, d in enumerate(devs):
        for tag in ("a", "b"):
            if devbuf[i][tag].value:
                F["zeMemFree"](d["ctx"], devbuf[i][tag])
        engines[i].destroy()
    for d in devs:
        if ok(d["ctx_rc"]) and d["ctx"].value:
            F["zeContextDestroy"](d["ctx"])
    print("done.")


if __name__ == "__main__":
    main()
