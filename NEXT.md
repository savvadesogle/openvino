# СЕССИЯ S12 — Stage 3.3 Hard: динамические state-шейпы, device-side коллективы/события, конкурентность · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §2 (численные инварианты:
     GQA 16 q-голов / 4 kv-головы / head_dim 256; размерности проекций кратны числу рангов),
     §4.2 (compile-time, срез до `propagate_constants`), §4.3 (per-rank state; out-proj K-split+allreduce),
     §5 (строка S12), §6 (тест-матрица), §7.2/§7.4 (события/транспорт, fp32-аккумуляция).
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая», блок **S12 стартует с**
     в записи S11, а также записи S10 (state-классификация, агрегация, REPLICATED-обоснование) и S08/S09
     (K-split + host reducer, `TPReplicatedAdd`).
  3) C:\openvino-tp\docs\notes\state-inventory.md — инвентарь состояний Qwen (S10 + §6 S11):
     точные conv/ssm-цепочки, head-оси, `?,4,?,256`/`?,8192,4`/`?,32,128,128`, вывод «реальный Qwen
     state = REPLICATED» и почему (coupled fused `in_proj`/`Concat`/`Loop`/`ScatterNDUpdate`).
  4) C:\openvino-tp\docs\notes\l0-spike.md — §8: единственный рабочий междевайсный путь = host-USM staging
     (~4–5 GB/s e2e); device-side швы/события ещё НЕ реализованы (S12 — впервые).
  5) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §о stateful/профиле/событиях, точка вставки пасса.
  6) C:\openvino-tp\docs\notes\cpu-tp-pattern.md и int4-asym-weights.md — паттерн TP и формы среза.
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S12 (таблица PLAN.md §5). Следующие этапы
  (S13 GenAI, S14+ scheduler/бенч/перф/стабилизация) ЗАПРЕЩЕНЫ. Остановка на GATE/блокере — нормальный
  результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC/кросс-shared USM NA).
  **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Device-side швы — предмет S12.
- S04 done: `TPCoordinator`/TP-ветка `compile_model`/свойства (`tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy`).
- S05 done: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы/состояние только rank0).
- S06 done: `TPShardTransform` классифицирует `MatMul`/`FullyConnected(Compressed)`; `tp_seams`.
- S07 done: реальный N-split (срез Constant по N-оси + bias) + host all-gather-эквивалент
  (`TPOutputSeam` ALL_GATHER, синтетический полный порт, конкатенация регионов); rank в пасс через
  internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks`.
- S08 done: K-split + host all-reduce(sum) — K-срез весов и активации (`ov::op::v8::Slice`, `axis=-1`);
  subtractive reducer `TPInferRequest::assemble_reduce_sum` (fp32); `TPOutputSeam.collective`+`replicated_adds`;
  RO-property `ov::intel_gpu::tp_output_seams`.
- S09 done: int4_ASYM — `match_quantized_weight` (dequant-подграф на операнде 1), N-split/K-split weight+scale+zp,
  sub-byte выравнивание (`slice_is_byte_aligned`), иначе → REPLICA.
- S10 done: per-rank KV + агрегация состояний — `analyze_state_variable`, `find_state_head_axis`,
  `TPStateSeam{variable_id, REPLICATED/HEAD_SHARDED, axis, full_dim, full_shape, reason}`,
  rt_info `intel_gpu_tp_state_seams`, RO-property `ov::intel_gpu::tp_state_seams`
  (`KIND:AXIS:FULL_DIM:VARIABLE_ID`); `TPAggregatedState : ov::IVariableState` (HEAD_SHARDED — конкатенация
  head-оси по рангам / scatter в `set_state`; REPLICATED — rank-0 авторитет + fan-out); generic
  `copy_axis_region`/`extract_axis_region`.
- **S11 done (2026-10-17, частичный GATE — сборка недоступна):** depthwise-conv-per-head примитив:
  `analyze_state_variable` обрабатывает head-local цепочку `ReadValue -> depthwise GroupConvolution ->
  (elementwise/broadcast) -> Assign`; `match_depthwise_head_conv` (filters `Constant` `[GROUPS,1,1,k]`,
  `GROUPS == data_channels == full_dim`), фильтры режутся по оси 0 (`slice_constant`+`replace_constant`,
  суффикс `_tp_headrank`), затем шардится переменная + `validate_nodes_and_infer_types()`.
  `find_state_head_axis` предпочитает **axis 1** (layout Qwen `[batch, heads, ...]`). Реальные Qwen
  conv/ssm-цепочки остаются **REPLICATED**. Тесты: `LinearAttentionConvStateHeadShardingMatchesSingleGpu`
  (heads=4), `LinearAttentionConvStateIndivisibleStaysReplicated` (heads=3) + билдер
  `make_linear_attn_conv_state_model(heads,taps)`.

## Задача S12 (полный объём этой сессии)
Stage 3.3 — **Hard: снять два оставшихся барьера, которые не дают реальному Qwen3.5-9B-int4 стать TP**:
(а) динамические шейпы состояний (все Qwen-состояния `?,...`); (б) отсутствие device-side/межрангового
транспорта для НАСТОЯЩЕГО per-rank шардинга (сейчас всё «host staging» / REPLICATED). Плюс конкурентность.
Цель GATE — паритет **Qwen3.5-9B-int4 TP2 vs 1-GPU (seq ~256)** или частичный GATE (код + план теста +
честная пометка в STATE), без регрессий S04–S11.

1. **Динамические state-шейпы.**
   - Все Qwen-состояния динамические (`?,4,?,256`, `?,8192,4`, `?,32,128,128`); сейчас
     `analyze_state_variable` отклоняет их на «dynamic state shape» → REPLICATED. Ввести шардинг по
     **статической head-оси** (обычно axis 1) при динамическом batch: `update_data_shape` сохраняет
     динамику в batch/seq, а head-ось делится. Проверить, что `validate_nodes_and_infer_types()` и
     потребители состояния допускают такую частично-динамическую форму.
   - Если потребитель требует полностью статический shape — обоснованно оставить REPLICATED и записать причину.

2. **Межранговый транспорт для настоящего шардинга (device-side коллективы/события).**
   - Реализовать реальный all-gather/all-reduce как минимум как корректный **host-USM staging**
     (G0): pinned-host буфер + per-device команды копирования, host-sync. Это заменит чисто
     host-конкатенацию/суммирование S07/S08 на границе узлов и позволит резать цепочки, где N-split узел
     питает следующий GEMM (inter-node all-gather), а также внутренний residual (K-split).
   - Рассмотреть device-side швы/события (`zeEventPool`, edge-от-`zeCommandQueueSynchronize`) как fast-path
     поверх staging; учесть находки S02/S03 (counter-based event + `zeEventHostSignal` = UNSUPPORTED;
     classic `zeEventCreate` из ctypes крашил драйвер — event pool аккуратно).
   - Оценить применимость peer-пути (`zeDeviceCanAccessPeer`) как runtime fast-path.
   - Атомарность: это НЕ полный device-side-стек ради перфа; приоритет — корректность и паритет.

3. **Конкурентность / порядок.**
   - Проверить корректность при асинхронном `infer()` (`create_async_infer_request`) и при нескольких
     одновременных запросах; per-rank sub-запросы сейчас выполняются последовательно (MVP S05).
     Задокументировать/защитить гонки в `replicate_inputs`/`assemble_outputs`/состояниях.
   - Проверить roundtrip состояния `query_state() → set_state() → infer()` для HEAD_SHARDED и REPLICATED.

4. **Не регрессировать S04–S11.**
   - Паритеты S05/S07/S08/S09/S10/S11, негативы S04, seams S06, TP-off — зелёные по коду/логике.
   - Форматы `tp_seams`/`tp_output_seams`/`tp_state_seams` стабильны (или расширяй согласованно, обновляя тесты).
   - Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S11), причина в debug-лог.

5. **GATE S12.**
   - Главный GATE: **Qwen3.5-9B-int4 TP2 паритет vs 1-GPU на seq ~256** (8 full_attention + 24 linear_attention).
     Если модель/сборка недоступны — **частичный GATE**: код + план теста + статическое ревью; честная
     пометка в STATE.
   - Минимальный функциональный тест (обязателен, если общая сборка недоступна): цепочка, где N-split узел
     питает следующий GEMM (нужен real inter-node all-gather) и/или внутренний residual для K-split;
     TP2 vs 1-GPU паритет на нескольких шагах. SKIP <2 GPU.
   - Мини-тест динамического состояния: динамический batch/seq + статическая head-ось → HEAD_SHARDED,
     паритет + корректная агрегация; негатив (голова не делится) → REPLICATED + паритет.

## GATE S12 (критерии завершения)
- Динамические state-шейпы шардируются по статической head-оси (или обоснованно REPLICATED).
- Межранговый транспорт реализован (host-USM staging как минимум) и применён к inter-node all-gather/all-reduce;
  device-side fast-path — если безопасно.
- Паритет Qwen3.5-9B-int4 TP2 (seq ~256) vs 1-GPU (или частичный GATE: код + план теста).
- S04–S11 не сломаны; форматы seams/state-форматы стабильны; TP-off == upstream.
- STATE.md: S12 done/блокер + артефакты + решения (транспорт, динамика, конкурентность) + заметки → S13.
- NEXT.md — полный самодостаточный промпт S13 (интеграция GenAI).
- Commit (сообщение: "S12: dynamic state shapes + inter-rank transport + concurrency") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S13+ (GenAI, scheduler, бенч, перф-оптимизации, стабилизация).
  S12 = динамика + межранговый транспорт + конкурентность.
- Не коммитить `.bin`/модельные файлы; секреты не писать в файлы/логи/историю; `.bin` (4.6 GB) не читать целиком.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18 (`docs/dev/coding_style.md`), в среде только clang-format 20.1.0
  → whole-file прогон даёт массовый нерелевантный churn. **Не запускать clang-format** на этих файлах; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать блокер
  в STATE и считать GATE частичным (код + план теста готовы).
