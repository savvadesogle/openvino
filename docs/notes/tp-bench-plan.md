# TP2 vs TP1/MULTI benchmark plan (S14)

Канон: PLAN.md §5 (S14), §6 (перф: TPOT/TTFT TP2/TP1 ≥ 1.5/1.3, память без дублирования весов),
§7.4 (fp32-аккумуляция). Харнесс: `tools/tp_bench/tp_bench.py`. Контекст транспорта — G0
(`docs/notes/l0-spike.md` §8, host-USM staging ~4–5 GB/s e2e).

## 1. Цель и метрики

Цель S14 — **spид-ап TP2 относительно 1-GPU** за счёт параллельного исполнения рангов (scheduler),
а не serial-режима S05–S13. Метрики (режим greedy, детерминированно):

- **TTFT** (time-to-first-token) — задержка prefill (первый токен): замер от старта генерации до
  первого нового токена. Отражает prefill-латентность (compute + первичные коллективы).
- **TPOT** (time-per-output-token) — средняя задержка последующих decode-шагов. Основной GATE-
  показатель (decode доминирует в интерактивной генерации).
- Производные: `TTFT_ratio = TTFT(TP1)/TTFT(TP2) ≥ 1.3`, `TPOT_ratio = TPOT(TP1)/TPOT(TP2) ≥ 1.5`.
- Память: TP2 не должен дублировать веса — срез на OV-уровне **до** `propagate_constants`/upload
  (PLAN §4.2, S07–S09). Проверяется индикативно (см. §5): суммарный RSS/VRAM двух рангов ≈ 1× веса
  для шардированных узлов; REPLICATED-узлы (реальный Qwen) дают ~1× на ранг (k× суммарно) — это
  известное ограничение S07–S13, не регресс.

## 2. Окружение

| Поле | Значение |
|---|---|
| Модель | `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml` (.bin ~4.6 GB — не читать целиком) |
| Устройства | 2× Intel Arc A770 (текущая машина) ИЛИ 2× Intel Battlemage (целевая, BMG — GATE §5) |
| Связь | PCIe; `zeDeviceCanAccessPeer` = false (S02) → коллективы = host-USM staging (G0) |
| Runtime | OpenVINO из этого форка (ветка `feature/gpu-tensor-parallel`), GenAI (`openvino_genai`) для реального TTFT/TPOT |
| ОС | Windows 11 |

Варианты компиляции (публичный API, см. `tp_bench.py::_device_configs`):

- **TP1**: `core.compile_model(model, "GPU")` — эталон 1-GPU.
- **TP2**: `core.compile_model(model, "GPU", {"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"})` —
  TP-ветка плагина (k рангов; S14 — параллельный scheduler).
- **MULTI** (референс, при наличии ≥2 GPU): `core.compile_model(model, "MULTI:GPU,GPU")` —
  штатный upstream-путь (независимые пайплайны), не TP.

## 3. Фиксированные параметры прогона

- Режим: **greedy** (`temperature=0`, `do_sample=False`, `num_beams=1`).
- `--warmup 2` (прогрев: компиляция/upload/кэш не входят в замер).
- `--iters 10` (серия из 10 независимых генераций, берётся медиана TTFT и TPOT).
- `--max-new-tokens 128` (decode-объём серии); `--prompt-len 128` (prefill-объём).
- Один прогон = одна генерация; для каждого варианта сначала warmup, затем iters замеров.

## 4. Команды

```powershell
# Полный набор (реальный TTFT/TPOT, нужен openvino_genai):
python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov `
    --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI `
    --out tp_bench_report.json

# При отсутствии GenAI — indicatively (decode-step proxy, только масштабирование):
python tools/tp_bench/tp_bench.py --variants TP1,TP2 --out tp_bench_proxy.json
```

Ручной сценарий (через GenAI LLMPipeline напрямую):

1. `LLMPipeline(model_dir, "GPU", **{ov.properties.device.properties("GPU"): {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}})` — TP2.
2. `LLMPipeline(model_dir, "GPU")` — TP1.
3. `GenerationConfig{max_new_tokens=N, do_sample=False, temperature=0}`; замер TTFT/TPOT
   (per-token timestamp streamer либо `perf_metrics` в GenAI).
4. Повторить для MULTI (если нужно) и сравнить отношения.

## 5. Профилирование узких мест (если GATE не достигнут)

Ожидаемые источники времени (для отчёта S15):

1. **REPLICATED-слои.** Реальный Qwen TP2 по-прежнему исполняет полный граф на каждом ранге
   (coupled fused `in_proj`/`Concat`/`Loop`/`ScatterNDUpdate`, нет in-graph коллективов, S07–S13).
   Тогда wall-time TP2 ≈ max(rank) + staging, а не compute/2 → TPOT_ratio ≈ 1.0, не 1.5. Нужен
   S15 (in-graph all-gather/all-reduce).
2. **host-USM staging (G0).** Межранговые коллективы идут dev→pinned host→dev, ~4–5 GB/s e2e,
   host-sync floor ~60–90 µs/вызов (S03). На большом числе слоёв/шагов это линейно суммируется.
   Профиль — time per collective × count.
3. **Prefill vs decode.** TTFT ограничен compute prefill (шардирование может дать >1.3× легче,
   чем decode); TPOT ограничен staging + serial-хвостами.
4. **Scheduler overhead.** Проверить, что `m_rank_executor` (k-stream pool) реально перекрывает
   ранги: сравнить serial (эмуляция) vs parallel scheduler на одном и том же графе.

Метод фиксации: `ov::InferRequest::get_profiling_info()` (в TP — с префиксами `rankN:`), плюс
внешний замер staging-полосы (`tools/l0_spike/l0_bench.py`).

## 6. Критерии приёмки (GATE S14)

- `TPOT(TP1)/TPOT(TP2) ≥ 1.5` **И** `TTFT(TP1)/TTFT(TP2) ≥ 1.3` на 2× BMG → GATE пройден.
- Иначе — **частичный GATE**: зафиксировать профиль (какая доля в REPLICATED, какая в staging),
  зафиксировать в STATE.md и передать в S15 (peer-копии, ring/tree allreduce).
- Среда сборки/железа недоступна (сабмодули пусты, нет 2 GPU/GenAI) → бенч не прогоняется,
  GATE частичный: готовы scheduler + обвязка + этот план.

## 7. Чек-лист запуска (когда доступно железо/сборка)

- [ ] Собрать OpenVINO из ветки (или взять собранный плагин), разложить `openvino` Python-пакет.
- [ ] Убедиться, что видны 2 GPU (`core.get_available_devices()` → `GPU.0`,`GPU.1`).
- [ ] Прогнать `tensor_parallel` функциональные тесты (паритет S05–S14 на Qwen3.5-9B-int4).
- [ ] Прогнать `tp_bench.py` полный набор; сохранить JSON.
- [ ] Снять профиль TP2 (`get_profiling_info` с `rankN:`): доля staging vs compute.
- [ ] Записать отношения в STATE.md (перф-журнал) → решение GATE S14.
