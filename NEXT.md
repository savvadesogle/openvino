# СЕССИЯ S15 — Stage 5.1 перф: peer-копии + ring/tree allreduce, prefill/decode отчёт (закрытие GATE TPOT ≥ 1.5×) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
Сначала полностью прочитай:
1. `C:\openvino-tp\PLAN.md` — канон (НЕ редактировать в сессии), особенно §4.4 (runtime:
   `TPCompiledModel`/`TPInferRequest`, выходы с ранга 0; коллективы — OpenCL-ядра через
   `custom_gpu_primitive`, fp32-аккум, staging peer-device при `zeDeviceCanAccessPeer`, иначе USM host),
   §5 (строка S15: «Stage 5.1 перф: peer-копии, ring/tree allreduce | prefill/decode отчёт»),
   §6 (перф: TPOT/TTFT TP2/TP1 ≥ 1.5/1.3), §7 (риски: fp16-суммы → fp32; фьюжны не пересекают швы).
2. `C:\openvino-tp\STATE.md` — особенно новый блок **S14 done** в «Заметки сессия → следующая»,
   а также S12 (`TPTransport` host-USM staging, конкурентность), S13 (GenAI-путь, REPLICATED-ожидание),
   S08/S09 (K-split + host reducer, fp32). «Блокеры» → S14 GATE частичный.
3. `C:\openvino-tp\docs\notes\tp-bench-plan.md` — **бенч-план S14** (метрики TTFT/TPOT, окружение
   2×A770 vs 2×BMG, команды `tools/tp_bench/tp_bench.py`, критерии, профиль узких мест).
4. `C:\openvino-tp\docs\notes\l0-spike.md` §3/§8 — capability-матрица (peer=false, IPC mem/event NA)
   и микро-полосы (host staging ~4–5 GB/s e2e, ~60–90 µs host-sync floor). **Сверить перед перфом.**
5. `C:\openvino-tp\docs\notes\gpu-plugin-arch.md` §3–§4 — очереди/события/`network::execute` (как
   публиковать/ждать в device-side швах), §6 (ограничители cldnn = 1 device).
6. `C:\openvino-tp\docs\notes\cpu-tp-pattern.md` §5 — эталон allgather-механики (двойной буфер +
   счётчик раунда) как идея пула staging-буферов по (seam, rank, iteration).

Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = **S15** (таблица PLAN.md §5). Следующий
этап (S16 стабилизация: blob-cache per-rank, docs, upstream-серии) ЗАПРЕЩЁН. Остановка на GATE/блокере —
нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC/кросс-shared USM NA);
  **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Peer — только fast-path hook.
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
- S14: **rank scheduler** — `TPInferRequest::schedule_ranks()` исполняет k рангов **параллельно** на
  общем k-stream `ov::threading::IStreamsExecutor` из `TPCompiledModel` (`m_rank_executor`,
  `get_rank_executor()`), через `ITaskExecutor::run_and_wait`; `replicate_inputs()` до fan-out,
  `assemble_outputs()`/агрегация после всех рангов (детерминизм); тест
  `SchedulerParallelRanksMatchSingleGpu`; бенч-харнесс `tools/tp_bench/tp_bench.py` + план
  `docs/notes/tp-bench-plan.md`. Реальный Qwen по-прежнему преимущественно **REPLICATED** →
  TPOT_ratio≈1.0; ≥1.5× не достигнут/не измерен.

## Задача S15 (полный объём этой сессии)

Stage 5.1 — **перф-оптимизация межранговых коллективов + prefill/decode отчёт; цель — закрыть GATE
TPOT ≥ 1.5× / TTFT ≥ 1.3×.** Точный объём:

1. **Peer-копии (fast-path) вместо host-staging, где доступно.**
   - В `TPTransport` (`tp_transport.{hpp,cpp}`) включи реальный PEER-путь при
     `zeDeviceCanAccessPeer(dev_i, dev_j) == true` (`TPTransportCapabilities::peer_available`):
     прямые device→device копии без pinned-host round-trip. На текущей машине (2×A770, S02) peer=false →
     путь не активируется; код/ветка должны быть готовы и не ломать HOST_STAGING fallback.
   - Сохрани единый интерфейс коллективов (`all_gather_axis`/`scatter_axis`/`all_reduce_sum_fp32`).
     Публичные свойства/seams не менять.
2. **Ring/tree allreduce.**
   - Замени host-суммирование в `all_reduce_sum_fp32` на **ring- или tree-allreduce** (PLAN §4.4:
     device-side ядра; на хосте — ring-схема через staging — приемлемо как MVP-перф). fp32-аккумуляция
     обязательна (PLAN §7.4), replicated bias/residual — ровно один раз (как S08).
   - Оцени алгоритмически: объём трафика ring vs текущий «все→rank0→все»; для k=2 показать выигрыш
     (одно плечо вместо двух).
3. **Staging-буферы per (seam, rank, iteration) — опционально** (если нужно для overlap):
   идея двойной буферизации + счётчика раунда из `cpu-tp-pattern.md` §5. Не обязателен, если
   single-request decode не страдает.
4. **prefill/decode отчёт.**
   - Прогон `tools/tp_bench/tp_bench.py` (или ручной сценарий из `tp-bench-plan.md`) на 2×A770 и/или
     2×BMG: TTFT/TPOT TP1/TP2/MULTI. Зафиксировать окружение, warmup/iters/seqlen/greedy.
   - Профиль TP2 (`get_profiling_info()` с префиксами `rankN:`): доля staging/коллективов vs compute.
   - Записать в STATE перф-журнал + решение GATE. Если ≥1.5× недостижимо из-за REPLICATED —
     честно зафиксировать потолок и что именно мешает (in-graph all-gather на coupled fused
     `in_proj`/`Concat`/`Loop` — вероятно, требует in-graph коллективов, частично вне S15).
5. **Не регрессировать S04–S14.**
   - Паритеты S05/S07–S14, негативы S04, seams S06, TP-off == upstream, GenAI-тесты S13,
     scheduler-тест S14.
   - Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S14).

## GATE S15 (критерии завершения)
- `TPTransport` умеет PEER fast-path (активируется по `zeDeviceCanAccessPeer`); HOST_STAGING не сломан.
- allreduce — ring/tree (не «все→rank0→все»); fp32-аккум сохранён; bias/residual один раз.
- Есть prefill/decode отчёт (реальный прогон ИЛИ план+профиль) с решением по TPOT/TTFT.
- Либо GATE TPOT ≥ 1.5× / TTFT ≥ 1.3× достигнут, **либо** частичный GATE с оценкой потолка
  (REPLICATED/staging) и передачей остатка в S16.
- STATE.md: S15 done/блокер + артефакты + перф-числа + заметки → S16.
- NEXT.md — полный самодостаточный промпт S16 (стабилизация: blob-cache per rank по (rank, device),
  docs/upstream-серии, матрица §6 зелень).
- Commit (сообщение: "S15: peer copies + ring allreduce + prefill/decode report") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S16 (blob-cache per-rank, upstream-серии, финальные docs-матрицы).
- Не коммитить `.bin`/модельные файлы; секреты не писать; `.bin` (4.6 GB) не читать целиком.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18, в среде только clang-format 20.1.0 →
  whole-file прогон даёт массовый churn. **Не запускать clang-format**; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или `docs/notes/`.
- Если сборка OpenVINO недоступна (сабмодули пусты) или нет 2 GPU — не изобретать обходные сборки;
  зафиксировать блокер в STATE и считать GATE частичным (peer/ring-код + профиль/план готовы).
