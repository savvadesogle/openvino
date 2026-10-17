# HETERO PIPELINE_PARALLEL — как устроен и почему не подходит для TP

Конспект S01. Пути/строки проверены на master `85720fe2` (2026-10-17).
Плагин: `src/plugins/hetero/src/`.

## 1. Конфигурация: только PP

`config.cpp:24-35`: при `ov::hint::model_distribution_policy` HETERO принимает
ровно `PIPELINE_PARALLEL`; любое другое значение → `OPENVINO_THROW`
(". HETERO plugin only support {PIPELINE_PARALLEL}"). `TENSOR_PARALLEL` для
HETERO отвергается на входе.

## 2. Как строится PP-пайплайн

- Разбиение модели: `Plugin::query_model_update`
  (`plugin.cpp:265-553`) — user-`affinity` из rt_info (`:45-51`) либо
  `query_model` на каждое устройство из `device_priorities`; узлы получают
  метки устройства в `SupportedOpsMap`.
- `SubgraphCollector` (`subgraph_collector.hpp:27-56`, `.cpp`):
  - граница подграфа = ребро между узлами с разной affinity
    (`subgraph_collector.cpp:129-150` — "any cross-affinity input edge is a
    boundary; Result nodes inherit producer affinity");
  - SCC-разрывы циклов `split_cyclic_dependencies()` `:151+`
    (вызывается из конструктора `:96`), присвоение id `collect_subgraphs_ids()`
    `:831+`, сборка субмоделей `run()` `:959+`;
  - в итоге `SubgraphsMappingInfo` (`subgraph_collector.hpp:18-26`):
    соответствие входов/выходов субмоделей (вход субмодели N+1 ← выход N).
- В модель вставляются `ov::hetero::op::DeviceSubgraph` (subgraph_collector.cpp:
  `:1401-1407`; op: `op/device_subgraph.hpp:28-56`, `get_affinity()`), HETERO
  `CompiledModel` компилирует каждую субмодель на своём устройстве.

## 3. Как исполняется: строго последовательно

- `sync_infer_request.cpp:127-132` (`InferRequest::infer`): цикл
  `for (auto&& request : m_subrequests) request->infer();` — субмодели
  исполняются **одна за другой**; промежуточные тензоры перекладываются между
  устройствами на границах. Никакого параллелизма по устройствам.
- `query_state()` `:114-125` — конкатенация состояний субмоделей.
- Профилирование `:134-143` — просто конкатенация.

## 4. Почему для TP непригоден напрямую

1. **Политика**: HETERO отвергает TENSOR_PARALLEL явно (config.cpp:24-35).
2. **Уровень разбиения**: PP режет по узлам (слоям/последовательностям) на
   границах affinity; TP режет **внутри узлов** (веса/головы одного и того же
   слоя) и требует, чтобы k рангов обрабатывали ОДИН слой одновременно.
3. **Исполнение**: PP — строгий последовательный цикл (sync_infer_request.cpp:
   127-132); TP требует одновременного запуска k рангов с обменом
   (publish→wait→assemble/allreduce) внутри слоя — этого PP-рантайм не умеет.
4. **Данные**: PP передаёт промежуточные тензоры между устройствами на
   границах subgraph'ов (host-тензоры), TP нужны device-side коллективы
   (IPC-события/staging, PLAN §4.4).

## 5. Что всё же полезно как референс

- `SubgraphsMappingInfo` (вход→предыдущий выход) — та же идея, что «швы» TPPlan,
  но на уровне тензоров слоя, а не субмоделей.
- `DeviceSubgraph`-изоляция: паттерн «субмодель с меткой устройства» — можно
  использовать для упаковки ранга (submodel per rank), если захочется
  переиспользовать часть tooling'а blob/сериализации.
- Роль HETERO в проекте: только подтверждает, что правильный образец для TP на
  GPU — **как у CPU**: один плагин-владелец владеет k подграфами/рангами
  (cpu-tp-pattern.md §3-4), а не композиция чужих плагинов.
