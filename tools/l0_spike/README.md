# tools/l0_spike — Level Zero capability + microbench spikes (S02/S03)

Read-only probes of inter-device Level Zero capabilities and data-transfer paths
on **Windows**, used as input for the TP collectives design (PLAN.md §4.4). They
do **not** touch `src/**` and require no build step (Python 3 + `ctypes` over the
system `ze_loader.dll`).

## Files

- `l0_capability.py` (S02) — capability probe: enumerates L0 GPU drivers/devices,
  prints the `zeDeviceCanAccessPeer` N×N matrix, P2P properties, and probes IPC
  memory and (counter-based) IPC events. Measures nothing.
- `l0_bench.py` (S03) — microbenchmark of the **available** inter-device paths
  (same-device baseline, host-USM staging device→host→device, shared USM). Sizes
  4 KB–128 MB, both directions, median/min/percentiles, effective GB/s + host-
  observed latency. Reuses types/loader helpers from `l0_capability.py`. Copies
  use regular command lists + in-order queues + `zeCommandQueueSynchronize`
  (no `zeEvent*`, which crashed the driver in S02).
- `l0_capability_output.txt`, `l0_bench_output.txt` — captured runs on this
  machine (2× Arc A770), referenced by `docs/notes/l0-spike.md`.

## Usage

```powershell
python l0_capability.py
python l0_bench.py
```

## Notes

- Struct layouts were transcribed from the installed Level Zero header
  (`C:\Program Files\LevelZeroSDK\1.33.1\include\level_zero\ze_api.h`); the
  `zex` counter-based event definitions come from the bundled reference headers
  `src/plugins/intel_gpu/src/runtime/ze/compute_runtime/zex_{common,event}.h`.
- `zex*` functions are not exported by `ze_loader.dll`; they are resolved through
  `zeDriverGetExtensionFunctionAddress`.
- The tool allocates 1 MiB device USM per GPU and frees it before exit; it makes
  no persistent system changes.
