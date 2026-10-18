# СЕССИЯ S13 — Stage 4.1 GenAI-форк: hint pass, CB на TP2, паритет генерации · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §4.1 (API/свойства:
     `model_distribution_policy{TENSOR_PARALLEL}`, `tp_ranks`, ошибки совместимости), §4.4 (runtime:
     `TPCompiledModel`/`TPInferRequest`, выходы с ранга 0), §5 (строки S13 и S14), §6 (тест-матрица),
     §7 (численные инварианты: fp32-аккумуляция, GQA 16 q-голов / 4 kv-головы / head_dim 256).
     Также §в начале (стр. ~79): «GenAI CB уже прокидывает `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}`».
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая» → блок **S13 стартует с**
     в записи S12, а также записи S12 (динамические state-шейпы, `TPTransport` host-USM staging, конкурентность),
     S10/S11 (state-классификация/агрегация, REPLICATED-обоснование) и S08/S09 (K-split + host reducer).
  3) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — архитектура GPU-плагина (точка вставки пасса, stateful,
     профилирование/события) и C:\openvino-tp\docs\notes\cpu-tp-pattern.md — эталон TP-паттерна CPU
     (агрегация состояний, биндинг выходов на ранг 0).
  4) C:\openvino-tp\docs\notes\int4-asym-weights.md и state-inventory.md — формы срезов и инвентарь состояний
     реального Qwen3.5-9B-int4 (`?,4,?,256` KV, `?,8192,4` conv, `?,32,128,128` ssm).
  5) C:\openvino-tp\docs\notes\l0-spike.md §8 — транспорт: единственный рабочий междевайсный путь = host-USM
     staging (G0, ~4–5 GB/s e2e).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S13 (таблица PLAN.md §5). Следующие этапы
  (S14 scheduler/бенч/перф, S15 peer/ring-allreduce, S16 стабилизация) ЗАПРЕЩЕНЫ. Остановка на GATE/блокере —
  нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC/кросс-shared USM NA);
  **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e); device-side события — краши/UNSUPPORTED.
- S04 done: `TPCoordinator`/TP-ветка `compile_model`; свойства `tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy` (в т.ч. ошибки «GPU.0 + TP», «PP на GPU», «TP на 1 GPU»).
- S05 done: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы/состояние только rank0);
  `create_infer_request()` → `create_async_infer_request<>()` на выделенном `CPUStreamsExecutor`.
- S06 done: `TPShardTransform` классифицирует `MatMul`/`FullyConnected(Compressed)`; RO-property `tp_seams`.
- S07 done: N-split (срез Constant по N-оси + bias) + all-gather-эквивалент (`TPOutputSeam` ALL_GATHER,
  синтетический полный порт, host-конкатенация); rank в пасс через `ov::intel_gpu::tp_rank`/`tp_num_ranks`.
- S08 done: K-split + all-reduce(sum) — K-срез весов и активации (`ov::op::v8::Slice`, axis=-1); subtractive
  reducer (fp32); `TPOutputSeam.collective`+`replicated_adds`; RO-property `ov::intel_gpu::tp_output_seams`.
- S09 done: int4_ASYM — `match_quantized_weight` (dequant-подграф на операнде 1), N-split/K-split weight+scale+zp,
  sub-byte выравнивание, иначе → REPLICA.
- S10 done: per-rank KV + агрегация состояний — `analyze_state_variable`, `find_state_head_axis`,
  `TPStateSeam{variable_id, REPLICATED/HEAD_SHARDED, axis, full_dim, full_shape, reason}`,
  RO-property `ov::intel_gpu::tp_state_seams` (`KIND:AXIS:FULL_DIM:VARIABLE_ID`); `TPAggregatedState`.
- S11 done: depthwise-conv-per-head примитив (filters `[GROUPS,1,1,k]` режутся по оси 0); head-ось — axis 1.
- **S12 done (2026-10-18, частичный GATE — сборка недоступна):** динамические state-шейпы
  (`TPStateSeam.full_shape` = `ov::PartialShape`; шардинг по **статической делимой** head-оси при динамическом
  batch/seq; иначе REPLICATED без throw); межранговый транспорт `TPTransport` (host-USM staging, G0):
  `all_gather_axis`/`scatter_axis`/`all_reduce_sum_fp32`, `tp_transport_capabilities()`; `assemble_outputs`/
  `assemble_reduce_sum`/`TPAggregatedState` ходят через транспорт; конкурентность (`m_infer_mutex`,
  `m_output_mutex`, независимые запросы). Тесты: `DynamicState*`, `StateRoundtrip*`, `ConcurrentRequestsMatchSingleGpu`.
- **Реальный Qwen3.5-9B-int4 по-прежнему преимущественно REPLICATED** (coupled fused `in_proj`/`Concat`/`Loop`/
  `ScatterNDUpdate`, нет true in-graph inter-node коллективов) — это ожидаемо для S13 (паритет генерации важнее,
  чем полный шардинг всех слоёв).

## Задача S13 (полный объём этой сессии)
Stage 4.1 — **GenAI-форк: hint pass + Continuous Batching на TP2; цель GATE — паритет генерации**
(детерминированная greedy-генерация TP2 vs 1-GPU на Qwen3.5-9B-int4). Это первый этап, где TP-путь GPU
используется «как в продакшене» — через OpenVINO GenAI (`LLMPipeline` / continuous-batching), а не через
прямой `core.compile_model`. Точный объём:

1. **Probe: как GenAI доводит `MODEL_DISTRIBUTION_POLICY=TENSOR_PARALLEL` до плагина.**
   - Найди в репозитории, где GenAI/OV-genai (или bundled tool) прокидывает
     `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` (PLAN §3, ~стр. 79:
     `tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp`).
   - Проверь, что свойство доходит до `Plugin::compile_model` как `ov::hint::model_distribution_policy` и
     попадает в TP-ветку `TPCoordinator`. Если GenAI передаёт строку в другом формате (property vs config) —
     зафиксируй разрыв и минимально закрой его на стороне GPU-плагина (без правки GenAI, если возможно).
   - Задокументируй путь в `docs/notes/gpu-plugin-arch.md` (новый §) и/или в STATE.

2. **Hint pass / совместимость compile-фрейма с GenAI.**
   - Убедись, что TP-компиляция не ломается на реальном Qwen3.5-9B-int4 (stateful, динамические шейпы,
     int4_ASYM-веса): узкие места — `TPShardTransform` на реальном графе, `validate_nodes_and_infer_types()`
     после `update_data_shape`, `build_output_ports()` для stateful-модели (`query_state`/`set_state` через
     `TPAggregatedState`).
   - Если GenAI требует чего-то, чего TP-путь не даёт (напр., `get_tensor` на промежуточных портах,
     unsupported-свойства, `ov::device::properties` на под-устройства), зафиксируй и, где дёшево, закрой.
   - Не регрессируй S04–S12: TP-off == upstream; форматы `tp_seams`/`tp_output_seams`/`tp_state_seams` стабильны.

3. **Continuous Batching на TP2 (паритет генерации).**
   - Прогони/спланируй детерминированную greedy-генерацию (temperature=0, фиксированный prompt) на
     Qwen3.5-9B-int4 через `LLMPipeline` или CB-бенчмарк с TP2 (2× Arc A770) и сравни с 1-GPU.
   - Критерий: совпадение сгенерированных токенов (greedy) на разумной длине; логиты — в пределах численного
     допуска (fp32-аккумуляция §7.4). Если паритет недостижим из-за REPLICATED-слоёв — зафиксируй, где именно
     расходится, и оцени вклад в STATE.
   - Проверь roundtrip `query_state()` → `set_state()` для stateful-модели в GenAI-контексте (KV + linear-attn).

4. **Не регрессировать S04–S12.**
   - Паритеты S05/S07/S08/S09/S10/S11/S12, негативы S04, seams S06, TP-off — зелёные по коду/логике.
   - Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S12), причина в debug-лог.

5. **GATE S13.**
   - Главный GATE: **паритет генерации** Qwen3.5-9B-int4 TP2 vs 1-GPU (greedy, LLMPipeline/CB). Если модель
     или сборка недоступны — **частичный GATE**: код/обвязка GenAI-пути + план теста + статическое ревью;
     честная пометка в STATE.
   - Минимальный функциональный тест (обязателен, если общая сборка недоступна): детерминированная генерация
     фиксированного prompt через публичный GenAI/CB API на TP2 и на 1-GPU, сравнение последовательностей
     токенов; SKIP <2 GPU. Если GenAI-пакет недоступен в среде — тест-план + ручной прогон-сценарий в docs/notes.

## GATE S13 (критерии завершения)
- `MODEL_DISTRIBUTION_POLICY=TENSOR_PARALLEL` из GenAI доходит до TP-ветки compile (путь задокументирован).
- Генерация на TP2 паритетна 1-GPU (greedy, фиксированный prompt) ИЛИ частичный GATE (обвязка + план теста).
- Состояние (KV + linear-attn) корректно roundtrip-ится через `query_state()`/`set_state()` в GenAI-контексте.
- S04–S12 не сломаны; форматы seams/state-форматы стабильны; TP-off == upstream.
- STATE.md: S13 done/блокер + артефакты + решения (GenAI-путь, ограничения реального Qwen) + заметки → S14.
- NEXT.md — полный самодостаточный промпт S14 (scheduler + бенч TP2 vs TP1/MULTI, GATE TPOT ≥ 1.5×).
- Commit (сообщение: "S13: GenAI integration (hint pass + CB on TP2)") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S14+ (scheduler, бенч-инфраструктура, перф-оптимизации, peer/ring-allreduce,
  стабилизация). S13 = GenAI-интеграция + паритет генерации.
- Не коммитить `.bin`/модельные файлы; секреты не писать в файлы/логи/историю; `.bin` (4.6 GB) не читать целиком.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18 (`docs/dev/coding_style.md`), в среде только clang-format 20.1.0
  → whole-file прогон даёт массовый нерелевантный churn. **Не запускать clang-format** на этих файлах; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать блокер
  в STATE и считать GATE частичным (код/обвязка + план теста готовы).
