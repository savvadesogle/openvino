# GPU-плагин: архитектура (база для TP-порта)

Конспект S01. Пути/строки проверены на master `85720fe2` (2026-10-17).
Дерево intel_gpu: `src/plugins/intel_gpu/src/` (далее — относительно него).

## 1. Свойства и точка входа

- Публичный API уже есть: `src/inference/include/openvino/runtime/properties.hpp:436-488`
  (см. cpu-tp-pattern.md §1). Python-bindings маршалят enum:
  `src/bindings/python/src/pyopenvino/utils/utils.cpp:560`.
- В intel_gpu **нет ни одного упоминания** `TENSOR_PARALLEL`/`model_distribution_policy`
  (grep по `src/plugins/intel_gpu`): свойство сейчас либо отвергается config-валидацией,
  либо молча игнорируется. Точка интеграции — `Plugin::compile_model`
  (`plugin/plugin.cpp:267-306`) + `ExecutionConfig`/`PluginConfig` (добавить
  внутренний `enable_tensor_parallel`, как у CPU).
- Перечисление устройств: `plugin/plugin.cpp:227-258` (`m_device_map` из
  `device_query.get_available_devices()`, дефолтные контексты на устройство);
  переименование id — `:342-351` (`m_default_device_id`).
- Заметка: `tools/continuous_batching` в ЭТОМ репо отсутствует (это репозиторий
  openvino.genai). В здешнем дереве строка `MODEL_DISTRIBUTION_POLICY` есть
  только в properties.hpp и pybind. Для S13 (GenAI-форк) смотреть отдельный репо.
  Путь свойства GenAI → TP-ветка плагина разобран в `docs/notes/genai-integration.md` §2
  (S13): строковый конфиг `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` конвертируется в
  `std::set<ModelDistributionPolicy>` через `ov::Any::as` и доходит до `compile_model_tensor_parallel`.

## 2. «1 CompiledModel = 1 device»

- `plugin/remote_context.cpp:57-72`: конструктор `RemoteContextImpl` —
  `OPENVINO_ASSERT(devices.size() == 1, "[GPU] Currently context can be created for single device only")` (`:59`); один `m_device` (`:60`).
- `plugin/compiled_model.cpp:56-75`: `CompiledModel` держит один `m_context` и
  один `m_wait_executor`; граф: `graph_base = Graph(model, context, config, 0)`,
  затем `num_streams` клонов `Graph(graph_base, n)` (`:70-75`) — **num_streams
  размножает граф на очередях ОДНОГО GPU** (CCS), а не на устройства.
- Shared-context creation (`remote_context.cpp:74-120`): поддержан только OCL
  context/queue sharing; ZE-interop контекста явно не поддержан
  (`:100-101` "Level Zero interoperability is not supported").
- Вывод: TP-путь = k независимых `(RemoteContextImpl, Graph/network)` — своя
  обёртка `TPCompiledModel`/`TPInferRequest`, существующий CompiledModel не
  масштабируется на несколько device (PLAN §4.4). Каждый per-rank граф
  строится и оптимизируется своим Engine'ом.

## 3. Компиляция: OV-пассы и cldnn-граф

- OV-трансформации: `plugin/transformations_pipeline.cpp`,
  `TransformationsPipeline::apply` начинается на `:555`.
  Место вставки TP-пассов (срез констант / маркировка швов) — **до фьюжнов и
  LowPrecision**, рядом с плагинными пассами, но раньше них по смыслу:
  SDPAToVLSDPA `:647`, TransposeMatMul `:681`, MarkDequantization `:683-686`,
  MoE-группа `:702-724`, GatedDeltaNetFusion `:727`. Поздние фьюжны, которые
  НЕ должны пересекать швы: RoPEFusion `:1642`, FuseRoPEInputDeinterleave
  `:1643`, FuseGatedMLP `:1651`, SwiGlu `:1653`.
- После OV-пассов `ov::Model` → cldnn `program` → opt-пассы →
  `program::post_optimize_graph` (`graph/program.cpp:607-644`), где
  выполняется `apply_opt_pass<propagate_constants>()` (`:628-629`,
  при `!is_internal && !partial_build`).
- `propagate_constants` (`graph/graph_optimizer/propagate_constants.cpp:238-280`):
  строит внутреннюю микро-сеть из константных подграфов, исполняет её на GPU
  (`net->execute({})` `:265`), результаты становятся `memory::ptr`-константами
  графа. **Это и есть «upload констант» на устройство.**
  ⇒ Срез весов ранга (`TPShardTransform`) обязан происходить **до**
  `propagate_constants` — на OV-уровне (в TransformationsPipeline), иначе
  полные веса будут материализованы на каждом GPU (PLAN §4.2, риск дублирования).

## 4. Runtime: очереди, события, барьеры

- `runtime/stream.cpp:11-16`: `get_expected_sync_method(config)` — profiling →
  `events`, out-of-order → `barriers`, иначе `none`.
- `graph/network.cpp:1048-1107` (`network::execute`): ждёт прошлое исполнение
  (`reset_execution(false)`), вызывает `execute_impl(dependencies)`.
  `execute_impl` `:1122-1182`: **последовательный цикл** по `_exec_order`
  (`:1141-1161`): `clear_events()` → при input-узлах `add_dep_events(events)`
  (`:1150-1152`) → `prepare_primitive()` → `execute()`; flush каждые 16 примов
  для dynamic (`:1137-1139`, `:1158-1160`); финальный `get_stream().flush()`
  (`:1176`); выходные события — из `out_event` при OOO/profiling (`:1099-1103`).
- L0-очереди: `runtime/ze/ze_stream.cpp:213-268` — immediate command list на
  CCS (`index = stream_id++ % num_ccs` `:218`), флаг `ZE_COMMAND_QUEUE_FLAG_IN_ORDER`
  для in-order (`:225`), copy-offload hint (`:229-236`), counter-based events
  включаются при `in_order && info.supports_counter_based_events` (`:244-250`;
  отключены на xe3/xe3p/unknown — GSD-13050).
  `enqueue_kernel()` `:310-334`: при `SyncMethods::events` деп-события
  пробрасываются в `zeCommandListAppendLaunchKernel` (dep_events),
  при `barriers` — `sync_events(deps)`.
- Детекция device-возможностей: `runtime/ze/ze_device.cpp:107-108`
  (`supports_extension(extensions, ZEX_COUNTER_BASED_EVENT_EXT_NAME, ...)`,
  `supports_cp_offload`), `:224-225`, `:257` (`num_ccs`).

## 5. L0 IPC: что уже есть в дереве (и что нет)

- **Есть** объявления IPC для counter-based events (не используются плагином):
  `runtime/ze/compute_runtime/zex_event.h:46-50`
  (`zexCounterBasedEventGetIpcHandle` / `OpenIpcHandle` / `CloseIpcHandle`),
  `runtime/ze/compute_runtime/zex_common.h:167-204`:
  `zex_ipc_counter_based_event_handle_t` (`:167-170`),
  флаг `ZEX_COUNTER_BASED_EVENT_FLAG_IPC = ZE_BIT(3)` (`:179`),
  ext `ZEX_counter_based_event` версии 1.0 (`:152-164`),
  `defaultIntelCounterBasedEventDesc` (`:206-214`).
- **Нет** в дереве: `zeDeviceCanAccessPeer`, `zeDeviceGetP2PProperties`,
  `zeMemGetIpcHandle`, `zeMemOpenIpcHandle` (grep по всему intel_gpu — ноль).
  ⇒ S02 берёт их из системных Level Zero headers/loader (вне репо).
- Detector: `runtime/ze/ze_device_detector.cpp:16-70` — sub-devices как «N.M»
  (`:54-65`), `create_device_list` `:72-100` — `zeInit(ZE_INIT_FLAG_GPU_ONLY)`
  (`:75`), `zeDriverGet`/`zeDeviceGet` (`:77-88`), фильтр `ZE_DEVICE_TYPE_GPU`
  (`:95`). Идентификаторы устройств далее → `m_device_map` плагина.
- Вывод для TP: коллективы/события рангов придётся строить поверх «сырого» L0
  (или через эти zex-хелперы), добавив уровень `TPCoordinator`; никакие
  существующие абстракции cldnn IPC не дают — их надо спроектировать
  (PLAN §4.4, S08/S12).

## 6. Ограничители и риски (следствия для дизайна)

1. cldnn-граф = 1 engine = 1 device: TP-файлы изолировать, минимальные
   касания существующего кода (PLAN §7.3).
2. `reset_execution`/`flush` в `network::execute` — синхронные хвосты исполнения
   одного графа; межранговые швы должны жить в событиях immune к этому контракту
   (TP-обёртка сама управляет enqueue k графов, выходы — rank0).
3. `propagate_constants` строит сеть на том engine, с которым компилируется
   program (per-rank) — слайс должен быть готов до этого (см. §3).
4. Counter-based events (в т.ч. IPC-вариант) включены не на всех arch
   (`ze_stream.cpp:246-250`) — на BMG (Xe2) должны быть доступны; проверить в S02.
5. Blob-cache (`plugin/compiled_model.cpp:77-140`): compat-descriptor проверяет
   runtime requirements; per-rank кэш надо различать по (rank, device) —
   учитывать в S16 (по PLAN), не сейчас.

## 7. GenAI-интеграция (S13)

- Полный разбор пути свойства `MODEL_DISTRIBUTION_POLICY=TENSOR_PARALLEL` от OpenVINO GenAI
  (LLMPipeline / continuous-batching) до TP-ветки `Plugin::compile_model`, анализ
  property-vs-config и совместимость compile-фрейма — в `docs/notes/genai-integration.md`.
- Кратко: GenAI передаёт строковый `ov::AnyMap` `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` в
  `core.compile_model(model,"GPU",config)`; `ov::PluginConfig::set_user_property` конвертирует строку
  в `std::set<ov::hint::ModelDistributionPolicy>` (`ov::Any::as`, `any.hpp:767-784`), GPU-плагин
  рекламирует свойство RW (`plugin.cpp:893`) и уходит в `compile_model_tensor_parallel`
  (`plugin.cpp:285`). Правок GenAI/плагина не требуется.
- TP-обёртка совместима с GenAI-потреблением stateful-модели: name-based `set_tensor/get_tensor`,
  `query_state()/set_state()` (KV/linear-attn). Эмуляционные тесты — `TensorParallelCompileGPU.
  GenAiStringConfigEnablesTensorParallel`, `GenAiStyleStatefulRoundtripMatchesSingleGpu`.
- Реальный Qwen3.5-9B-int4 на TP2 остаётся преимущественно REPLICATED (S07–S12) → генерация
  должна быть численно паритетна 1-GPU; in-graph межранговые коллективы — вне S13.
