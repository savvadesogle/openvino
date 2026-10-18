# tools/tp_bench — TP2 vs TP1/MULTI TTFT/TPOT benchmark (S14)

Benchmark harness for the S14 GATE (PLAN.md §5/§6): measure **TTFT** and **TPOT** for
`Qwen3.5-9B-int4-ov` through the public OpenVINO API and compare tensor parallel (TP2)
against a single GPU (TP1) and, optionally, an upstream `MULTI` reference.

- **TP2** — `core.compile_model(model, "GPU", {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"})`
  (k ranks across all enumerated GPUs, GPU-plugin TP branch).
- **TP1** — `core.compile_model(model, "GPU")` (single-device reference).
- **MULTI** — `core.compile_model(model, "MULTI:GPU,GPU")` (upstream independent pipelines;
  a *reference*, not a TP path).

GATE: `TPOT(TP1)/TPOT(TP2) >= 1.5` and `TTFT(TP1)/TTFT(TP2) >= 1.3` on 2x BMG, with no weight
duplication (S07-S09 slice before upload).

## Files

- `tp_bench.py` — the harness (Python 3, `openvino` required; `openvino_genai` preferred).
- `README.md` — this file.

## Usage

```powershell
# Real TTFT/TPOT (preferred) — requires the openvino_genai package:
python tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov --max-new-tokens 128 --warmup 2 --iters 10

# Subset of variants / JSON report:
python tp_bench.py --variants TP1,TP2 --out tp_bench_tp2_vs_tp1.json
```

If `openvino_genai` is not importable, the harness falls back to an indicative **decode-step
proxy** (`ov.decode_step_proxy`): it times a single stateful inference step on synthetic inputs.
That proxy is enough to compare scaling across variants but does **not** yield a real TPOT/TTFT
ratio, so the GATE is only evaluated when GenAI metrics are present.

## Notes

- Warmup / iteration count / greedy decoding are fixed on the command line and recorded in the
  JSON report. `temperature=0`, `do_sample=False`, `num_beams=1`.
- Real `Qwen3.5-9B-int4` TP2 is still dominated by **REPLICATED** layers (S07-S13), so both GPUs
  execute the full graph and the gain comes only from overlapping them (S14 scheduler) minus the
  host-USM staging collectives (G0). The 1.5x target needs the S15 in-graph collective work.
- The full manual scenario, environment matrix and acceptance criteria are in
  `docs/notes/tp-bench-plan.md`.
