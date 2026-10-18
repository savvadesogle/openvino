# S16: перф-матрица TP1 / TP2 / MULTI (аналитика + чек-лист замеров)

Канон: PLAN.md §6 (`TPOT(TP1)/TPOT(TP2) ≥ 1.5`, `TTFT(TP1)/TTFT(TP2) ≥ 1.3`), §7.4 (fp32).
Источники: `docs/notes/s15-perf-report.md` (трафик/крит. путь ring, prefill/decode-профиль),
`docs/notes/tp-bench-plan.md` (метрики/среда/команды), `tools/tp_bench/tp_bench.py`.

## 0. Статус

**Замеры выполнены на целевом железе (2× Intel Arc A770), 2026-10-18.** Собрана OpenVINO
`2026.5.0-26-2941d4f40c9-feature/gpu-tensor-parallel` (Release; ядро + GPU-плагин + Python-биндинги
+ `ov_gpu_func_tests`). `openvino_genai` в дереве отсутствует → перф снят в **indicative
`decode_step_proxy`** (один decode-step, синтетические входы, seq=1); **GATE-отношения на proxy не
считаются** (см. §5). Функциональные `*TensorParallel*`: **24 / 35 passed, 11 failed** — из них
**10 падений** вызваны pre-existing ограничениями GPU-плагина на stateful-моделях, которые
использует тест-набор (воспроизводятся и на **stock upstream-сборке `2026.5.0-22967`**, т.е. НЕ
регресс TP), и **1** — реальный TP-баг (потеря имён выходных тензоров, §6). Детали — §2, §2.1.

## 1. Варианты

| Variant | Как компилируется | Что делает |
|---|---|---|
| **TP1** | `core.compile_model(model, "GPU")` (1 device) | baseline; полный граф на одном GPU |
| **TP2** | `core.compile_model(model, "GPU", {model_distribution_policy:{TENSOR_PARALLEL}})` | 2 ранга; срез шардируемых узлов, коллективы на инференсе |
| **MULTI** | `core.compile_model(model, "MULTI:GPU,GPU")` | верхнеуровневый MULTI (для сравнения парадигм) |

Метрики: `TTFT` (prefill, мс), `TPOT` (decode, мс/токен), память (VRAM/ранг), отношения
`TTFT(TP1)/TTFT(TP2)`, `TPOT(TP1)/TPOT(TP2)`.

## 2. Аналитическая матрица (ожидание, Qwen3.5-9B-int4, 2× A770)

| Ячейка | Ожидание | Обоснование |
|---|---|---|
| TTFT(TP2) vs TP1 | **≈1.0×** (иногда хуже) | Qwen преимущественно REPLICATED → оба GPU считают полный граф; ring ускоряет только K-split-коллективы |
| TPOT(TP2) vs TP1 | **≈1.0×** | decode = `L` мелких коллективов; при доминировании REPLICATED `wall ≈ max(rank) + staging` |
| TTFT(MULTI) | ≈TT1 | MULTI без TP-шардинга (проверка парадигмы) |
| Память/ранг | **1×** на шардированных, **1×** полного на REPLICATED | срез весов до upload (N/K-split); REPLICATED дублируется на каждом ранге |
| Детерминизм/корректность | **паритет** с TP1 | сборка выхода fp32, реплики учитываются один раз (S08) |

**Вывод:** целевые GATE-отношения `TPOT ≥ 1.5× / TTFT ≥ 1.3×` на текущем Qwen **не достигаются**:
нужны (а) in-graph межранговые коллективы + шардинг реальных слоёв (снятие REPLICATED),
и/или (б) peer-копии (на A770 `zeDeviceCanAccessPeer` = false).

## 2.1 Фактические замеры (2× Arc A770, 2026-10-18)

Компиляция реального `openvino_language_model.xml` на `GPU` / `GPU+TENSOR_PARALLEL` проходит.
Замер — `tools/tp_bench/tp_bench.py` в **indicative proxy** (без GenAI): медиана времени одного
decode-step (`request.infer()`, синтетические входы, seq=1, warmup=2, iters=10).

| Variant | Device / props | step_s (медиана) | Отношение | Источник |
|---|---|---|---|---|
| **TP1** | `GPU` (1 устройство) | **0.0512 s** | — | `decode_step_proxy` |
| **TP2** | `GPU` + `MODEL_DISTRIBUTION_POLICY=TENSOR_PARALLEL` | **0.0643 s** | `TP1/TP2 = 0.80` | `decode_step_proxy` |
| **MULTI** | `MULTI:GPU,GPU` | — | — | **не поддержано**: `AUTO cumulative model doesn't support stateful model` |

Профиль **TP2** (`get_profiling_info`, префиксы `rankN:`, тот же LM, 1 decode-step):
`rank0 = 20.29 ms`, `rank1 = 20.17 ms` node real-time (оба ранга считают **полный** граф — REPLICATED);
wall per-step ≈ 64 ms → **≈24 ms/шаг невычислительного оверхеда** (host-USM staging / репликация
входов / синхронизация двух устройств). Коллективов на реальном Qwen почти нет (состояния REPLICATED),
поэтому TP2 не ускоряется, а **слегка замедляется**. Это подтверждает аналитический потолок §2/§3.

## 2.2 Окружение

| Параметр | Значение |
|---|---|
| Ветка / коммит | `feature/gpu-tensor-parallel`, `2941d4f4` (+ незакоммиченные build-fix правки — вошли в этот commit) |
| Версия OpenVINO | `2026.5.0-26-2941d4f40c9-feature/gpu-tensor-parallel` (Release) |
| Сборка | MSVC 19.44 (`cl` VS 2022 17.14 Community), CMake 4.0.3, Ninja 1.13.2, Python 3.11.9 |
| ОС | Windows (x64) |
| CPU | Intel Xeon E5-2699 v3 |
| GPU | 2× Intel Arc A770, драйвер 32.0.101.9150; `zeDeviceCanAccessPeer` = false → host-USM staging (G0) |
| Тест-модель | `models/Qwen3.5-9B-int4-ov` (LM `openvino_language_model.xml`) |
| GenAI | отсутствует (`openvino.genai` нет) → реальные TTFT/TPOT не сняты |
| Параметры бенча | `--warmup 2 --iters 10 --max-new-tokens 128 --variants TP1,TP2,MULTI` (greedy) |

## 3. Что даёт каждый перф-механизм (аналитически)

| Механизм | Слой | Эффект | Условие активации |
|---|---|---|---|
| Scheduler (S14) | executor | снимает serial-регресс (≈2× при REPLICATED) | всегда при TP2 |
| Ring all-reduce (S15) | транспорт | крит. путь k=2: `S` vs `2S` байт на коллектив | `S % k == 0`, иначе HOST_SUM |
| Peer fast-path (S15) | транспорт | исключает host round-trip/floor | `zeDeviceCanAccessPeer` все пары = true |
| In-graph коллективы | граф | **не реализовано** — снимет REPLICATED | follow-up |

## 4. Чек-лист замеров (на целевой машине)

Среда: 2× Intel Arc (A770 или BMG), Windows/Linux, релизная сборка OpenVINO + `openvino_genai`,
модель `models/Qwen3.5-9B-int4-ov`.

- [x] `core.get_available_devices()` содержит ≥2 `GPU.N`. → `['GPU.0','GPU.1']`.
- [x] Прогнать `*TensorParallel*` (S05–S16): **24/35 passed, 11 failed** (см. §0/§6).
- [x] `python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov
      --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json`
      — выполнен; без `openvino_genai` помечено как **indicative proxy**, GATE-отношения не считались.
      (Харнесс-фикс для dynamic-shape LM — в этом commit.)
- [x] Заполнить §2 фактами: `TTFT`, `TPOT`, отношения, память/ранг → см. §2.1 (proxy step; TTFT/VRAM N/A).
- [x] Профиль TP2 (`get_profiling_info`, префиксы `rank0:/rank1:`) — см. §2.1.
- [ ] При доступном peer-доступе: зарегистрировать `TPTransport::set_peer_region_copy` и повторить
      (ожидается рост TPOT-ratio). → **на A770 peer недоступен** (`zeDeviceCanAccessPeer` = false),
      ветка неактивна, host-staging (G0) — зафиксировано как ожидаемое.
- [x] Записать результаты в STATE.md → решение GATE (см. §5).

## 5. Критерии GATE (PLAN §6)

| Гейт | Порог | Факт (2× A770, 2026-10-18) |
|---|---|---|
| `TPOT(TP1)/TPOT(TP2)` | ≥ 1.5 | **не пройден / не измерим на GenAI**; indicative proxy: `TP1/TP2 = 0.80` (TP2 медленнее) |
| `TTFT(TP1)/TTFT(TP2)` | ≥ 1.3 | **не измерим** (нет `openvino_genai`; proxy даёт только per-step) |
| Паритет TP2 vs TP1 (корректность) | числ. совпадение в tol. | **частично**: S05/S06/S07/S08/S09/S14/S15 + S12-concurrency + S16 export/import (несostatey) — **зелёные**; stateful-паритет (S10–S12/S13/S16-stateful) **не проверен** — тест-модели не компилируются на GPU (pre-existing) |

**Текущий вердикт: GATE частичный.**
- Ядро шардинга (N-split/K-split/int4 + ring all-reduce + scheduler + per-rank кэш) **подтверждено на
  железе** (паритет TP2 vs 1-GPU).
- Перф-порог **не достигнут** и по аналитике недостижим без in-graph коллективов и peer (потолок
  REPLICATED + host-staging G0). Фактически TP2 ≈ 0.8× TP1 (indicative).
- Stateful-паритет **не подтверждён**: 10 тестов падают из-за pre-existing ограничений GPU-плагина
  (не TP-регресс); 1 реальный TP-баг — §6.

## 6. Известные проблемы (найденные на железе)

### 6.1 Реальный TP-баг: потеря имён выходных тензоров (S13 GenAI)
`GenAiStyleStatefulRoundtripMatchesSingleGpu` падает на `get_tensor("logits")`:
`Port for tensor name logits was not found`. При TP-компиляции `TPCompiledModel::output(0).get_names()`
пуст, тогда как plain single-GPU сохраняет имена. Причина — `TPCompiledModel::build_output_ports()`
(`tp_compiled_model.cpp:90-96`) создаёт **синтетический** full-shape порт (ALL_GATHER-выход, в т.ч. у
head-sharded состояния) и не переносит `port.get_names()` на тензор нового `Result`. Итог: name-based
binding (`get_tensor(name)`, GenAI LLMPipeline) для all-gathered выходов ломается. Минимальный репро —
`docs/notes/…` / STATE «Заметки»; фикс — однострочный (propagate names + friendly name), **вне
test-only сессии** → follow-up (S17, `NEXT-FEATURES.md`).

### 6.2 Pre-existing ограничения GPU-плагина (НЕ TP-регресс; воспроизведено на stock `2026.5.0-22967`)
Тест-модели TP-набора используют паттерны, которые плагин не компилирует **и без TP**:
1. **Stateful `ReadValue` + `Add` при rank-3 тeнзоре состояния со «средней» осью >2** (напр.
   `[1,4,2]+[1,1,2]`): `Argument shapes are inconsistent` (eltwise shape-inference в
   `program_builder.cpp:169`). Падают `StateHeadShardingMatchesSingleGpu`,
   `StateNonBroadcastOperandStaysReplicated`, `DefaultCompileHasNoStateSeams`,
   `StateRoundtripHeadShardedMatchesSingleGpu`, `StateRoundtripReplicatedMatchesSingleGpu`,
   `S16ExportImportStatefulRoundtripMatchesSingleGpu` (+ динамический вариант
   `DynamicStateHeadShardingMatchesSingleGpu` — падение на `infer`).
2. **Stateful rank-3 depthwise `GroupConvolution`** (данные `[1,4,3]`, фильтры `[4,1,1,3]`):
   плагин поднимает ранг до 5 → `Data batch and filters rank do not match`. Падают
   `LinearAttentionConvStateHeadShardingMatchesSingleGpu`, `LinearAttentionConvStateIndivisibleStaysReplicated`.
   (Плоский `Parameter` rank-3 GroupConvolution компилируется — проблема именно в stateful-ветке.)
3. **Динамическое состояние**: `query_state()` отдаёт batch-ось `0` вместо `1`
   (`[0,3,2]` vs `[1,3,2]`) — `DynamicStateIndivisibleStaysReplicated` (ожидание теста не совпадает
   с поведением плагина; воспроизведено и на stock).

**Итог по корректности:** паритет TP-ядра (S05–S09) подтверждён; stateful-паритет требует либо
починки тест-моделей под плагин (другой набор форм / явный init), либо фикса плагина, и это вне
test-only сессии.
