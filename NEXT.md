# СЕССИЯ S16 — Stage 5.2 стабилизация: per-rank blob-cache, финальные доки, upstream-серия · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
Сначала полностью прочитай:
1. `C:\openvino-tp\PLAN.md` — канон (НЕ редактировать в сессии), особенно §4.2 (per-rank blob-cache/
   upload до `propagate_constants`), §4.4 (runtime: `TPCompiledModel`/`TPInferRequest`, коллективы),
   §5 (строка S16: «Stage 5.2 стабилизация | blob-cache per-rank, docs, upstream-серии»),
   §6 (перф-матрица: TPOT/TTFT TP2/TP1 ≥ 1.5/1.3, память без дублирования весов),
   §7 (риски: fp16-суммы → fp32; фьюжны не пересекают швы), §8 (upstream-план/серия PR).
2. `C:\openvino-tp\STATE.md` — новый блок **S15 done** в «Заметки сессия → следующая», плюс S14
   (scheduler + бенч-обвязка), S12 (`TPTransport`, host-USM staging, конкурентность), S13 (GenAI-путь),
   S08/S09 (K-split + host reducer, fp32), «Гейт-решения» S01–S15, «Блокеры» (S15 GATE частичный).
3. `C:\openvino-tp\docs\notes\s15-perf-report.md` — трафик-модель (ring), prefill/decode-профиль,
   чек-лист прогона, GATE-решение S15 (**читай перед S16**).
4. `C:\openvino-tp\docs\notes\tp-bench-plan.md` — бенч-план S14 (метрики TTFT/TPOT, окружение, команды,
   критерии, профиль узких мест); `tools/tp_bench/tp_bench.py` + `README.md`.
5. `C:\openvino-tp\docs\notes\gpu-plugin-arch.md` §5–§6 — how-to точной вставки в upload/blob-cache
   (кэш моделей/`CreateModelEngine`, per-rank `ExecutionConfig`, ограничители cldnn = 1 device).
6. `C:\openvino-tp\docs\notes\cpu-tp-pattern.md` §5 — эталон allgather-механики (двойной буфер +
   счётчик раунда) для пула staging-буферов по (seam, rank, iteration), если понадобится.

Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = **S16** (таблица PLAN.md §5). Более
поздние этапы (upstream-PR вне описанной серии, новые фичи вне стабилизации) ЗАПРЕЩЕНЫ. Остановка
на GATE/блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC/кросс-shared USM NA);
  **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e, host-sync floor ~60–90 µs).
- S04–S05: `TPCoordinator`/TP-ветка `compile_model`; `TPInferRequest` (k per-rank sub-запросов,
  реплика входов, выходы/состояние rank0); `create_infer_request()` → async-обёртка.
- S06: `TPShardTransform` классифицирует seams; RO-property `tp_seams`.
- S07: N-split + all-gather-эквивалент (host-конкатенация, `TPOutputSeam` ALL_GATHER).
- S08: K-split + all-reduce(sum) — K-срез весов/активации, subtractive reducer (fp32), bias в reducer.
- S09: int4_ASYM — `match_quantized_weight`, N/K-срез weight+scale+zp, sub-byte выравнивание.
- S10/S11: per-rank KV + агрегация состояний (`TPStateSeam`), depthwise-conv-per-head.
- S12: динамические state-шейпы; `TPTransport` (host-USM staging: `all_gather_axis`/`scatter_axis`/
  `all_reduce_sum_fp32`); конкурентность (`m_infer_mutex`/`m_output_mutex`).
- S13: GenAI-путь (`docs/notes/genai-integration.md`); строковый GenAI-конфиг доходит до TP-ветки.
- S14: **rank scheduler** — `schedule_ranks()` (k-stream `m_rank_executor`), `replicate_inputs()` до
  fan-out, `assemble_outputs()` после; бенч-харнесс `tools/tp_bench/tp_bench.py` + план.
- S15: **peer-probe + ring allreduce**:
  - `zeDeviceCanAccessPeer` в weak-symbols (`zero_api.hpp`); `cldnn::gpu_devices_peer_accessible(size_t)`
    (`device_query.hpp` + `src/runtime/tp_peer.cpp`) — консервативная проба, не бросает;
  - `tp_transport_capabilities(size_t)` (кэш по k, `kind=PEER` при peer-доступе); `TPTransport::PeerRegionCopy`
    + `set_peer_region_copy()` (на A770 peer=false → ветка инертна, host-staging G0);
  - `all_reduce_sum_fp32` → ring reduce-scatter + all-gather (`ring_all_reduce_fp32`, fp32, `S%k==0`,
    иначе `HOST_SUM`); `last_all_reduce_stats()`;
  - тест `KsplitRingAllReduceDeterministicAcrossRuns`; отчёт `docs/notes/s15-perf-report.md`.
  - Реальный Qwen по-прежнему преимущественно **REPLICATED** → перф-GATE не подтверждён (см. блокер).

## Задача S16 (полный объём этой сессии)

Stage 5.2 — **стабилизация и подготовка к upstream.** Точный объём:

1. **Per-rank blob-cache.**
   - Кэш скомпилированных блобов должен быть **per (rank, device)**: ранги одного TP-группы не должны
     переиспользовать чужой блоб (у каждого ранга — свои срезы весов и своя `ExecutionConfig`
     `tp_rank`/`tp_num_ranks`, S07). Проверить и, где нужно, расширить ключ кэша
     (`CreateModelEngine`/`m_model_cache` в `plugin.cpp`/`compiled_model.cpp`) полями rank/num_ranks
     (или гарантировать раздельные блобы для TP-ранга). Зафиксировать решение в STATE (без изменения
     публичного API и форматов существующих свойств).
   - Не сломать обычный (не-TP) путь и MULTI: не-TP блоб-кэш — как раньше.
2. **Документация.**
   - Актуализировать/дописать доки в `docs/notes/` по итогам S04–S15: единая карта артефактов
     (файлы, свойства, seam-форматы, тесты), ограничения (REPLICATED, host-staging G0, peer простаивает),
     краткое «как запустить и что ожидать». Не создавать дублирующих доков; дельты — в существующие.
3. **Upstream-серия.**
   - Подготовить план разбиения изменений S04–S16 на серию независимо ревьюируемых коммитов/PR
     (PLAN §8): выделить связные слайсы (свойства+coordinator → shard-transform → collectives/transport →
     scheduler → state → GenAI → perf), для каждого — заголовок/объём/зависимости/тесты. Только план
     (документ), **без push и без создания PR** (remote-политика).
4. **Перф-матрица §6 (насколько доступно).**
   - Прогнать `tools/tp_bench/tp_bench.py` на 2×A770 и/или 2×BMG, если среда позволяет; зафиксировать
     TTFT/TPOT TP1/TP2/MULTI, окружение, warmup/iters/seqlen/greedy; записать в STATE перф-журнал.
   - Если сборка/железо недоступны — не изобретать обходные сборки: зафиксировать блокер, GATE частичный
     (матрица — по аналитике S15 + чек-лист `s15-perf-report.md` §7).
5. **Не регрессировать S04–S15.**
   - Паритеты S05/S07–S15, негативы S04, seams S06, TP-off == upstream, GenAI-тесты S13, scheduler S14,
     ring-детерминизм S15.
   - Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S15).

## GATE S16 (критерии завершения)
- Blob-cache корректен для TP (per-rank): ранги не делят чужой блоб; не-TP путь не изменён.
- Доки по S04–S16 актуальны (карта артефактов + ограничения + запуск/ожидания).
- Готов план upstream-серии (слайсы/заголовки/зависимости/тесты) — документ, без push/PR.
- Есть перф-матрица (реальный прогон ЛИБО аналитика+чек-лист) с решением по §6.
- Либо GATE §6 зелёный, **либо** частичный GATE с честной оценкой потолка (REPLICATED/staging) и
  явным списком остатка.
- STATE.md: S16 done/блокер + артефакты + перф-числа + заметки (финал проекта).
- NEXT.md — финальный самодостаточный промпт (если проект завершается — явное «проект завершён» +
  чек-лист для возобновления на железе).
- Commit (сообщение: "S16: per-rank blob-cache + docs + upstream series plan") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch); PR не создавать.
- Атомарность этапа: НЕ реализовывать фичи вне стабилизации (in-graph коллективы и т.п. — отдельные
  будущие этапы, только в плане upstream-серии).
- Не коммитить `.bin`/модельные файлы; секреты не писать; `.bin` (4.6 GB) не читать целиком.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18, в среде только clang-format 20.1.0 →
  whole-file прогон даёт массовый churn. **Не запускать clang-format**; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или `docs/notes/`.
- Если сборка OpenVINO недоступна (сабмодули пусты) или нет 2 GPU — не изобретать обходные сборки;
  зафиксировать блокер в STATE и считать GATE частичным.
