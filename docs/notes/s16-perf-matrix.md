# S16: перф-матрица TP1 / TP2 / MULTI (аналитика + чек-лист замеров)

Канон: PLAN.md §6 (`TPOT(TP1)/TPOT(TP2) ≥ 1.5`, `TTFT(TP1)/TTFT(TP2) ≥ 1.3`), §7.4 (fp32).
Источники: `docs/notes/s15-perf-report.md` (трафик/крит. путь ring, prefill/decode-профиль),
`docs/notes/tp-bench-plan.md` (метрики/среда/команды), `tools/tp_bench/tp_bench.py`.

## 0. Статус

**Фактические замеры отсутствуют** — сборка OpenVINO недоступна (сабмодули пусты), 2× GPU и
`openvino_genai` на сборочной машине нет (STATE.md, «Блокеры»). Ниже — аналитическая матрица и
готовый чек-лист. Колонка «измерено» заполняется на целевой машине.

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

- [ ] `core.get_available_devices()` содержит ≥2 `GPU.N`.
- [ ] Прогнать `*TensorParallel*` (S05–S16) — паритет/детерминизм/кеш-раундтрип зелёные.
- [ ] `python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov
      --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json`
      (greedy; при отсутствии `openvino_genai` — помечается indicative proxy, GATE-отношения не считать).
- [ ] Заполнить §2 фактами: `TTFT`, `TPOT`, отношения, память/ранг.
- [ ] Профиль TP2 (`get_profiling_info`, префиксы `rankN:`) — доля staging vs compute.
- [ ] При доступном peer-доступе: зарегистрировать `TPTransport::set_peer_region_copy` и повторить
      (ожидается рост TPOT-ratio).
- [ ] Записать результаты в STATE.md → финальное решение GATE.

## 5. Критерии GATE (PLAN §6)

| Гейт | Порог | Факт (заполнить) |
|---|---|---|
| `TPOT(TP1)/TPOT(TP2)` | ≥ 1.5 | н/д (не измерено) |
| `TTFT(TP1)/TTFT(TP2)` | ≥ 1.3 | н/д (не измерено) |
| Паритет TP2 vs TP1 (корректность) | полное совпадение (tolerance числ.) | подтверждается тестами (статически) |

**Текущий вердикт: GATE частичный** — корректность/структура готовы, перф-порог не подтверждён
измерением и, по аналитике, требует follow-up (in-graph коллективы / peer).
