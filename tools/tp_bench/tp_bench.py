#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# [TP-GPU, S14] TP2 vs TP1/MULTI TTFT/TPOT benchmark harness.
#
# Measures prefill latency (TTFT) and per-token decode latency (TPOT) for the
# Qwen3.5-9B-int4-ov model through the *public* OpenVINO runtime API:
#   - TP2  : core.compile_model(model, "GPU", {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"})
#   - TP1  : core.compile_model(model, "GPU")               (single device reference)
#   - MULTI: core.compile_model(model, "MULTI:GPU,GPU")     (upstream pipeline reference, optional)
#
# GATE (PLAN.md §6): TPOT(TP2)/TPOT(TP1) >= 1.5x and TTFT(TP2)/TTFT(TP1) >= 1.3x on 2x BMG.
# The real Qwen3.5-9B-int4 graph is still mostly REPLICATED in TP2 (S07-S13), so the
# collectives (host-USM staging, G0) dominate and the target may not be met until S15.
#
# See docs/notes/tp-bench-plan.md for the full run scenario and acceptance criteria.

import argparse
import json
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(msg, flush=True)


def _device_configs() -> Dict[str, Tuple[str, dict]]:
    """Device-name + compile-property for each measured variant.

    TP2 requests tensor parallelism across all enumerated GPUs. MULTI is an upstream
    reference (independent pipelines per device, no sharding) and only valid with >=2 GPUs.
    """
    tp_policy = {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}
    return {
        "TP1": ("GPU", {}),
        "TP2": ("GPU", tp_policy),
        "MULTI": ("MULTI:GPU,GPU", {}),
    }


def _count_gpu_devices(core) -> int:
    count = 0
    for device in core.get_available_devices():
        if device.startswith("GPU.") and device.count(".") == 1:
            count += 1
    return count


def _measure_genai(model_dir: str, device: str, props: dict, max_new_tokens: int,
                   warmup: int, iters: int) -> Optional[dict]:
    """Measure TTFT/TPOT with OpenVINO GenAI's LLMPipeline when available.

    LLMPipeline drives the stateful decoder exactly like production; TTFT is the first
    token latency and TPOT the mean latency of the remaining tokens. Greedy decoding.
    """
    try:
        import openvino_genai as ov_genai
    except Exception:
        return None

    # LLMPipeline forwards extra keyword properties into Core::compile_model; passing the
    # TENSOR_PARALLEL policy verbatim mirrors the production GenAI path (docs/notes/genai-integration.md).
    pipe = ov_genai.LLMPipeline(model_dir, device, **props) if props else \
        ov_genai.LLMPipeline(model_dir, device)
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.do_sample = False
    gen_cfg.temperature = 0.0
    try:
        gen_cfg.num_beams = 1
    except Exception:
        pass

    prompt = "What is OpenVINO?"

    def _one_run() -> Tuple[float, float]:
        streamer = ov_genai.StreamerBase() if hasattr(ov_genai, "StreamerBase") else None
        # Prefer explicit token timestamps when the Python API exposes a streamer; otherwise
        # fall back to wall-clock around generate() (TTFT/TPOT collapse to one number).
        if streamer is not None:
            timestamps: List[float] = []

            class _S(ov_genai.StreamerBase):
                def put(self, token_id):
                    timestamps.append(time.perf_counter())
                    return ov_genai.StreamerBase.PUT_OK if hasattr(ov_genai.StreamerBase, "PUT_OK") else 0

                def end(self):
                    return True

            t0 = time.perf_counter()
            pipe.generate(prompt, gen_cfg, _S())
            t_end = time.perf_counter()
            if len(timestamps) >= 1:
                ttft = timestamps[0] - t0
                if len(timestamps) >= 2:
                    otp = [(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
                    tpot = statistics.mean(otp)
                else:
                    tpot = t_end - timestamps[0]
            else:
                ttft = t_end - t0
                tpot = ttft
            return ttft, tpot

        t0 = time.perf_counter()
        pipe.generate(prompt, gen_cfg)
        elapsed = time.perf_counter() - t0
        # Without per-token timing, approximate: one prefill + (n-1) decode steps.
        tpot = elapsed / max(1, max_new_tokens)
        return elapsed, tpot

    for _ in range(max(0, warmup)):
        _one_run()
    ttfts, tpots = [], []
    for _ in range(max(1, iters)):
        ttft, tpot = _one_run()
        ttfts.append(ttft)
        tpots.append(tpot)
    return {
        "ttft_s": statistics.median(ttfts),
        "tpot_s": statistics.median(tpots),
        "samples": len(ttfts),
        "source": "openvino_genai.LLMPipeline",
    }


def _measure_step_proxy(core, model_path: str, device: str, props: dict,
                        warmup: int, iters: int) -> dict:
    """Fallback when OpenVINO GenAI is unavailable: time a stateful decode-step proxy.

    This does NOT produce real TPOT/TTFT (needs tokenizer + greedy loop); it measures the
    per-step wall time of the compiled language model on synthetic inputs sized to the
    declared ports, which is enough to compare scaling across TP1/TP2/MULTI before S15.
    """
    import numpy as np

    model = core.read_model(model_path)
    compiled = core.compile_model(model, device, props)
    request = compiled.create_infer_request()

    inputs = compiled.inputs
    input_tensors = {}
    for port in inputs:
        shape = [d.get_length() if hasattr(d, "get_length") else int(d) for d in port.get_partial_shape()]
        shape = [int(s) if isinstance(s, int) and s > 0 else 1 for s in shape]
        et = port.get_element_type
        np_dtype = np.int64 if str(et) in ("i64",) else (np.int32 if str(et) in ("i32",) else np.float32)
        input_tensors[port] = np.zeros(shape, dtype=np_dtype)

    def _one_step() -> float:
        request.set_input_tensors(input_tensors)
        t0 = time.perf_counter()
        request.infer()
        return time.perf_counter() - t0

    for _ in range(max(0, warmup)):
        _one_step()
    samples = [_one_step() for _ in range(max(1, iters))]
    return {
        "step_s": statistics.median(samples),
        "samples": len(samples),
        "source": "ov.decode_step_proxy (no GenAI; indicative only)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TP2 vs TP1/MULTI TTFT/TPOT benchmark (S14)")
    parser.add_argument("--model-dir", default="models/Qwen3.5-9B-int4-ov",
                        help="HF snapshot directory (config.json + openvino_* models)")
    parser.add_argument("--lm", default=None,
                        help="explicit openvino_language_model.xml path (default: <model-dir>/openvino_language_model.xml)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--variants", default="TP1,TP2,MULTI",
                        help="comma-separated subset of TP1,TP2,MULTI")
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    args = parser.parse_args()

    try:
        import openvino as ov
    except Exception as exc:  # pragma: no cover
        _log(f"openvino Python package is required: {exc}")
        return 2

    core = ov.Core()
    num_gpus = _count_gpu_devices(core)
    _log(f"available devices: {core.get_available_devices()}")
    _log(f"GPU devices: {num_gpus}")

    configs = _device_configs()
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    lm_path = args.lm or f"{args.model_dir}/openvino_language_model.xml"

    report = {
        "model": args.model_dir,
        "lm": lm_path,
        "num_gpus": num_gpus,
        "warmup": args.warmup,
        "iters": args.iters,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "greedy": True,
        "variants": {},
    }

    for name in requested:
        if name not in configs:
            _log(f"unknown variant '{name}', skipping")
            continue
        device, props = configs[name]
        if name in ("TP2", "MULTI") and num_gpus < 2:
            _log(f"{name} requires >=2 GPUs (have {num_gpus}); skipping")
            continue
        _log(f"[{name}] device={device} props={props}")
        result = _measure_genai(args.model_dir, device, props, args.max_new_tokens,
                                args.warmup, args.iters)
        if result is None:
            try:
                result = _measure_step_proxy(core, lm_path, device, props, args.warmup, args.iters)
            except Exception as exc:
                _log(f"[{name}] measurement failed: {exc}")
                report["variants"][name] = {"error": str(exc)}
                continue
        report["variants"][name] = result
        _log(f"[{name}] {json.dumps(result)}")

    # GATE evaluation (only when real TTFT/TPOT are available for TP1 and TP2).
    v = report["variants"]
    if "TP1" in v and "TP2" in v and "tpot_s" in v.get("TP1", {}) and "tpot_s" in v.get("TP2", {}):
        tpot_ratio = v["TP1"]["tpot_s"] / v["TP2"]["tpot_s"]
        ttft_ratio = v["TP1"]["ttft_s"] / v["TP2"]["ttft_s"]
        report["gate"] = {
            "tpot_ratio_tp2_over_tp1": tpot_ratio,
            "ttft_ratio_tp2_over_tp1": ttft_ratio,
            "tpot_pass": tpot_ratio >= 1.5,
            "ttft_pass": ttft_ratio >= 1.3,
        }
        _log(f"GATE TPOT ratio={tpot_ratio:.3f} (>=1.5?), TTFT ratio={ttft_ratio:.3f} (>=1.3?)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        _log(f"report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
