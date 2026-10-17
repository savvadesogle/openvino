# СЕССИЯ S11 — Stage 3.2 linear_attention: delta-состояния per head, conv, out-proj · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §2 (численные инварианты:
     GQA 16 q-голов / 4 kv-головы / head_dim 256; размерности проеций кратны числу рангов), §4.2
     (compile-time, срез до `propagate_constants`), §4.3 (**`linear_attention`: in-proj N-split по
     key/value-головам; depthwise conv per head; рекуррентное delta-состояние — rank-local; out-proj —
     K-split + allreduce**), §5 (строка S11), §6 (тест-матрица), §7.
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая», блоки **S10→S11**
     (state-классификация, head-local анализ, REPLICATED для реального Qwen) и S08/S09 (K-split + reducer).
  3) C:\openvino-tp\docs\notes\state-inventory.md — инвентарь состояний Qwen (S10): `past.conv.N`
     `?,8192,4`, `past.ssm.N` `?,32,128,128`, linear-attention ветка (`Loop`+`ScatterUpdate`+
     `GroupConvolution`), вывод «реальный Qwen state = REPLICATED», заметка про S11.
  4) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — паттерн TP (что режется/реплицируется; per-rank state).
  5) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3 (pass до `propagate_constants`), §о stateful/профиле.
  6) C:\openvino-tp\docs\notes\int4-asym-weights.md — форма dequant-подграфа и правила среза (S09).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S11 (таблица PLAN.md §5). Следующие этапы
  (S12 device-side коллективы/события, S13 GenAI, S14+ бенч/перф/стабилизация) ЗАПРЕЩЕНЫ. Остановка на
  GATE/блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC NA). **G0 = host-USM
  staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Device-side швы — S12.
- S04 done: `TPCoordinator`/TP-ветка `compile_model`/свойства (`tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy`).
- S05 done: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы/состояние только rank0).
- S06 done: `TPShardTransform` классифицирует `MatMul`/`FullyConnected(Compressed)`:
  нет константных весов → `REPLICA`; выход фида в `ov::op::v1::Add` → `K_SPLIT`(`ALL_REDUCE_SUM`);
  иначе → `N_SPLIT`(`ALL_GATHER`). `tp_seams` = `KIND:COLLECTIVE:TYPE:NAME`.
- S07 done: реальный N-split (срез Constant по N-оси + bias input 2) + host all-gather-эквивалент
  (`TPOutputSeam` ALL_GATHER, синтетический полный порт, конкатенация регионов). Rank прокинут в пасс через
  internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks` → `register_pass<TPShardTransform>(rank, num_ranks)`.
- S08 done: K-split + all-reduce(sum) — K-срез весов и активации (`ov::op::v8::Slice`, `axis=-1`);
  host-side subtractive reducer `TPInferRequest::assemble_reduce_sum` (fp32-сумма частичных,
  `- (num_ranks-1)*term` для replicated bias/residual `TPReplicatedAdd` CONSTANT/INPUT);
  `TPOutputSeam.collective`+`replicated_adds`; RO-property `ov::intel_gpu::tp_output_seams`
  (формат `COLLECTIVE:producer:split_axis:shard_dim_size:repl_adds`).
- S09 done: int4_ASYM — `match_quantized_weight` распознаёт dequant-подграф на операнде 1 GEMM; N-split и
  K-split weight+scale+zp (+target `Reshape`); sub-byte выравнивание (`slice_is_byte_aligned`), иначе → REPLICA.
- **S10 done (главный контекст для S11):** per-rank KV + агрегация состояний:
  - Инвентарь: 64 `ReadValue`/`Assign` (opset6), 0 Stateful; 8 `ScaledDotProductAttention` (full attn),
    24 × (`Loop` + `ScatterUpdate` + `GroupConvolution`) = linear attn. Формы состояния — в state-inventory.md.
  - `TPShardTransform` классифицирует каждую переменную: `analyze_state_variable` (статическая форма,
    числовой тип, делимая head-ось `find_state_head_axis` = ось `rank-2`/первая делимая, цепочка
    `ReadValue -> (element-wise/broadcast) -> Assign` целиком head-local, init_value ReadValue тоже broadcast)
    → `HEAD_SHARDED` (`var->update_data_shape` + `validate_nodes_and_infer_types`), иначе `REPLICATED` (без throw).
  - `TPStateSeam{variable_id, TPStateShardKind(REPLICATED/HEAD_SHARDED), axis, full_dim, full_shape, reason}`
    в `TPPlan.state_seams`; rt_info `intel_gpu_tp_state_seams`; RO-property `ov::intel_gpu::tp_state_seams`
    (формат `KIND:AXIS:FULL_DIM:VARIABLE_ID`).
  - `TPInferRequest::query_state()` строит `TPAggregatedState : ov::IVariableState` (совпадение по имени =
    variable_id): HEAD_SHARDED — конкатенация head-оси по рангам / scatter (`extract_axis_region`) в `set_state`;
    REPLICATED — rank-0 авторитетный + fan-out на `reset`/`set_state`.
  - `copy_last_axis_region` заменён generic `copy_axis_region`/`extract_axis_region`; `build_output_ports`/
    `assemble_outputs` — all-gather по произвольной оси; выходы от HEAD_SHARDED-состояния получают свой шов.
  - **Вывод S10**: реальные Qwen KV/conv/ssm-цепочки не head-local (SDPA/Loop/Concat/GroupConvolution —
    не element-wise) → в S10 всё `REPLICATED`. Per-rank KV для Qwen требует S12.

## Задача S11 (полный объём этой сессии)
Stage 3.2 — **linear_attention: per-head delta-состояние, depthwise conv, out-proj**. Учить TP-путь
корректно обрабатывать 24 linear-attention слоя Qwen3.5 (per-head свёрточное/рекуррентное состояние) и
агрегировать его. Цель GATE — паритет **32 слоя, seq ~256** vs 1-GPU (или частичный GATE: код + план теста +
статическое ревью, честная пометка в STATE).

1. **Инвентаризация linear-attention ветки.**
   - По `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml` (только чтение; `.bin` не коммитить)
     разобрать подграф linear-attention: вход in-proj (q/k/v или fused), depthwise `GroupConvolution`,
     `Loop`/`ScatterUpdate`, выход out-proj. Определить, где именно живёт delta-состояние (`past.conv.N`
     `?,8192,4`, `past.ssm.N` `?,32,128,128`) и как оно связано с головами.
   - Зафиксировать оси head-размерности для conv-состояния и ssm-состояния, кратность числа голов числу
     рангов. Дописать/обновить `docs/notes/state-inventory.md` (раздел S11).

2. **Per-head шардинг linear-attention состояния (rank-local).**
   - PLAN §4.3: in-proj N-split по key/value-головам; depthwise conv per head; рекуррентное delta-состояние
     — rank-local (аналог KV); out-proj — K-split + allreduce.
   - Расширить анализ состояний S10 (`analyze_state_variable`/`find_state_head_axis`) так, чтобы
     linear-attention-состояния `past.conv.*`/`past.ssm.*` могли становиться `HEAD_SHARDED`, если их
     head-цепочка локальна для ранга (depthwise conv/покомпонентная рекурренция head-local), иначе
     `REPLICATED` без throw (как S10). Обосновать выбор по каждой переменной.
   - Согласовать со срезами проекций S07/S09 (N-split in-proj по головам, K-split out-proj) и с GQA-ratio,
     если применимо. НЕ реализовывать device-side коллективы/события (S12): агрегация состояния — host-side.

3. **Агрегация состояния в `TPInferRequest`.**
   - Убедиться, что `TPAggregatedState`/`query_state()` корректно обрабатывает linear-attention-состояния
     (конкатенация head-оси при HEAD_SHARDED; rank-0 при REPLICATED) — переиспользовать обобщённые
     `copy_axis_region`/`extract_axis_region` из S10.
   - Проверить roundtrip состояния: `query_state()` → `set_state()` → повторный `infer()` даёт тот же
     результат (для KV из S10 и для linear attention).

4. **Не регрессировать S04–S10.**
   - Паритеты S05/S07/S08/S09/S10, S04-негативы, S06-seams, TP-off — зелёные по коду/логике.
   - `tp_seams`/`tp_output_seams`/`tp_state_seams` форматы стабильны (или расширяй согласованно, обновляя тесты).
   - Fallback при неделимости/небезопасном потребителе/невыровненности → репликация без throw (как S07–S10),
     с записью причины в debug-лог.

5. **GATE S11.**
   - Главный GATE: **Qwen3.5-9B-int4 TP2 паритет vs 1-GPU на seq ~256** (все 32 слоя: 8 full_attention +
     24 linear_attention). Если модель/сборка недоступны — **частичный GATE**: код + план теста +
     статическая проверка ревью; честная пометка в STATE. Модель: `models/Qwen3.5-9B-int4-ov/`.
   - Минимальный функциональный тест (обязателен, если общая сборка недоступна): линейно-внимание-подобная
     мини-модель с per-head состоянием (depthwise conv + рекуррентное состояние), TP2 vs 1-GPU паритет на
     нескольких шагах, проверка per-rank state и агрегации. SKIP <2 GPU.
   - Негатив: число голов/каналов не кратно числу рангов → репликация (без throw) и паритет.

## GATE S11 (критерии завершения)
- linear-attention-состояния (`past.conv.*`, `past.ssm.*`) per-head шардируются (или обоснованно
  реплицируются) + `TPInferRequest` агрегирует состояние.
- Паритет Qwen3.5-9B-int4 TP2 (seq ~256) vs 1-GPU (или частичный GATE: код + план теста).
- S04–S10 не сломаны; `TPPlan`/`tp_seams`/`tp_output_seams`/`tp_state_seams` стабильны; TP-off == upstream.
- STATE.md: S11 done/блокер + артефакты + решения (форма state, шардинг голов, агрегация) + заметки → S12.
- NEXT.md — полный самодостаточный промпт S12 (device-side коллективы/события/конкурентность).
- Commit (сообщение: "S11: linear-attention per-head state sharding + aggregation") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S12+ (device-side коллективы/события как прод-путь, GenAI, бенч).
  S11 = только linear_attention per-head состояние + агрегация.
- Не коммитить `.bin`/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18 (`docs/dev/coding_style.md`), в среде только clang-format 20.1.0
  → whole-file прогон даёт массовый нерелевантный churn. **Не запускать clang-format** на этих файлах; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать блокер
  в STATE и считать GATE частичным (код + план теста готовы).
