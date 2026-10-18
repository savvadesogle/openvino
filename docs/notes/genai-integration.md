# OpenVINO GenAI integration: how `MODEL_DISTRIBUTION_POLICY=TENSOR_PARALLEL` reaches the GPU plugin

Конспект S13. База — локальный форк на master `85720fe2` (2026-10-17); код OpenVINO GenAI
(репо `openvino.genai`) в этом дереве **отсутствует** (находка S01), поэтому путь ниже разобран
по контракту OV Core + публичному API и по ссылке PLAN.md §3 (~стр. 79). Компиляция в среде
недоступна (сабмодули пусты, «Блокеры» STATE) → это часть частичного GATE S13.

## 1. Где GenAI задаёт свойство (probe)

- PLAN.md §3: GenAI continuous-batching бенчмарк прокидывает
  `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` —
  `tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp` **в репо openvino.genai**
  (не в openvino). В дереве openvino строка `MODEL_DISTRIBUTION_POLICY` встречается только в
  `src/inference/.../properties.hpp` и pybind — CB-тула здесь нет.
- Форма: **строковый** `AnyMap` (ключ = имя свойства, значение = строка `"TENSOR_PARALLEL"`),
  который GenAI (LLMPipeline / ContinuousBatchingPipeline) передаёт как обычный `ov::AnyMap`
  в `ov::Core::compile_model(model, "GPU", config)`.

## 2. Путь свойства до TP-ветки плагина (проверен по коду)

1. `ov::Core::compile_model` (`src/inference/src/cpp/core.cpp:113`) → `CoreImpl::compile_model`
   (`src/inference/src/dev/core_impl.cpp:1143`).
2. `parse_device_name_into_config` разбирает device name и config (в т.ч. `ov::device::properties`),
   `get_supported_property` (`core_impl.cpp:1408`) фильтрует ключи по `ov::supported_properties`.
   GPU рекламирует `ov::hint::model_distribution_policy` как **RW**
   (`src/plugins/intel_gpu/src/plugin/plugin.cpp:893`), поэтому ключ не отбрасывается, значение
   (строка) сохраняется as-is.
3. `plugin.compile_model(model, parsed.m_config)` (не proxy) → `Plugin::compile_model`
   (`src/plugins/intel_gpu/src/plugin/plugin.cpp:267`).
4. `ExecutionConfig::set_user_property(orig_config, RELEASE)` (через базу `ov::PluginConfig`,
   `src/inference/src/dev/plugin_config.cpp:81`) валидирует значение:
   `option->is_valid_value(val)` → `val.as<std::set<ov::hint::ModelDistributionPolicy>>()`.
   **Строка → set<enum> конвертируется**: `std::set<...>` — `Readable`
   (`src/core/include/openvino/core/any.hpp:214`, `Read<std::set<K,...>>`), а
   `ModelDistributionPolicy` `Istreamable` (operator>> в `properties.hpp:456`). Конвертация
   выполняется Any-путём `as_impl` (`any.hpp:767-784`) при хранении `std::string`.
5. `Plugin::is_tensor_parallel_requested(config)` (`tp_coordinator.cpp:40`) проверяет
   `get_enable_tensor_parallel() || model_distribution_policy.count(TENSOR_PARALLEL)` →
   **TP-ветка** `compile_model_tensor_parallel` (`plugin.cpp:285`).
6. `TPCoordinator` штампует `tp_rank`/`tp_num_ranks` в per-rank config; `TransformationsPipeline::apply`
   регистрирует `TPShardTransform(config.get_tp_rank(), config.get_tp_num_ranks())` только при
   TP (`transformations_pipeline.cpp:653-661`), до фьюжнов и до `propagate_constants`.
7. Пользователю: `TPCompiledModel::get_property(ov::intel_gpu::tp_ranks)` → `{0,1}`;
   `get_property(ov::hint::model_distribution_policy)` → `{TENSOR_PARALLEL}` (RO, `tp_compiled_model.cpp:156`).

**Вывод (расхождение property vs config): закрывать на стороне плагина не требуется.**
И строковая (`"TENSOR_PARALLEL"`), и типизированная (`model_distribution_policy({TENSOR_PARALLEL})`)
формы доходят до TP-ветки. Строковая форма подтверждена тестом
`TensorParallelCompileGPU.GenAiStringConfigEnablesTensorParallel` (эмуляция ровно GenAI-конфига).

## 3. Что TP-путь даёт GenAI (совместимость compile-фрейма)

GenAI (LLMPipeline, stateful-путь) использует публичный `ov::InferRequest` API. TP-обёртка его
покрывает:

- `TPCompiledModel::inputs()/outputs()` — rank-0 порты (для ALL_GATHER-выходов — синтетический
  полной формы порт с сохранённым friendly name, `tp_compiled_model.cpp:45`).
- `create_infer_request()` → async-обёртка над `TPInferRequest` (`tp_compiled_model.cpp:100`).
- **Name-based** `set_tensor("...")/get_tensor("...")` — работает через `getPort` над
  `get_inputs()/get_outputs()` (`src/inference/src/cpp/infer_request.cpp:74,188`), т.е. над
  портами TP-модели. Проверено тестом `GenAiStyleStatefulRoundtripMatchesSingleGpu` (имена
  `input_ids`/`logits`).
- `query_state()`/`set_state()` — `TPAggregatedState` (KV/linear-attn): HEAD_SHARDED агрегируется
  по head-оси, REPLICATED — rank-0-авторитет с fan-out на все ранги
  (`tp_infer_request.cpp:69-153,391`). Это ровно то, что делает GenAI, гоняя KV между шагами.
- Прочие свойства делегируются rank-0 (`tp_compiled_model.cpp:160`).

## 4. Ограничение реального Qwen3.5-9B-int4

Реальная модель по-прежнему преимущественно **REPLICATED** (coupled fused `in_proj`/`Concat`/`Loop`/
`ScatterNDUpdate`, нет true in-graph inter-node коллективов — S07–S12). Для GenAI-паритета это
значит: ранги исполняют одинаковые полные графы, выходы/состояние берутся с ранга 0 → **численный
результат обязан совпасть с 1-GPU детерминированно** (greedy). Полный TP-спид-ап требует
межранговых коллективов внутри графа — вне S13 (S14+ perf, S15 peer/ring).

## 5. План теста (минимальный функциональный, при недоступной сборке)

Fallback-тест (в `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`, SKIP <2 GPU):
детерминированная/пошаговая генерация с фикс. входом через публичный API на TP2 (строковый
GenAI-конфиг + name-based IO + `query_state/set_state`) vs 1-GPU, сравнение выходов.

Ручной сценарий (когда доступны сборка OpenVINO + OpenVINO GenAI + 2× Arc):

1. `ov::Core core;` `core.compile_model("models/Qwen3.5-9B-int4-ov/openvino_language_model.xml", "GPU",
   {{"MODEL_DISTRIBUTION_POLICY","TENSOR_PARALLEL"}})` — TP2; отдельный compile без свойства — 1-GPU.
2. `ov::genai::LLMPipeline` с `temperature=0`, `do_sample=false`, фиксированные `input_ids`/seed;
   `GenerationConfig{max_new_tokens=N}`.
3. Сравнить последовательности токенов (полное совпадение) и логиты (допуск fp32, PLAN §7.4).
4. Roundtrip: между шагами `query_state()`→`set_state()` (KV + linear-attn) — паритет сохраняется.
5. Если GenAI-пакет недоступен в среде — использовать CB-бенчмарк из openvino.genai напрямую.

## 6. Риски/заметки

- `parse_device_name_into_config` может вызывать `get_supported_property`: при добавлении TP-свойств
  держать их в `get_supported_properties` (`plugin.cpp:893-894`) — уже сделано.
- GenAI на GPU обычно компилирует с `GPU` (не `GPU.N`) — TP это требует
  (`tp_coordinator.cpp:53`), а явный `GPU.0` отклоняется (`plugin.cpp` негативы S04).
- Blob-cache per-rank (S16) должен различать `(rank, device)`, иначе TP-кэш смешает ранги.
