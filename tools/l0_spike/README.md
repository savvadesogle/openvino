# tools/l0_spike — S02 Level Zero capability spike

Read-only probe of inter-device Level Zero capabilities on **Windows**, used as
input for the TP collectives design (PLAN.md §4.4). It does **not** measure
performance (that is S03) and does **not** touch `src/**`.

## Files

- `l0_capability.py` — the probe. Python 3 + `ctypes` over the system
  `ze_loader.dll` (no build step). Enumerates L0 GPU drivers/devices, prints the
  `zeDeviceCanAccessPeer` N×N matrix, P2P properties, and probes IPC memory and
  (counter-based) IPC events.
- `l0_capability_output.txt` — captured run on this machine (2× Arc A770),
  referenced by `docs/notes/l0-spike.md`.

## Usage

```powershell
python l0_capability.py
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
