# ПРОМПТ-ПЛАН: новые этапы TP-GPU (S17–S20) — follow-up после S16

> Мастер-план для последующих сессий. Проект S01–S16 завершён (корректность/структура/кеш готовы,
> GATE §6 частичный). Здесь — новые этапы, которые внедряются **по одному за сессию**.
> Перед стартом этапа прочитать `STATE.md` (последний блок «Заметки»), `docs/notes/tp-artifacts.md`,
> `docs/notes/upstream-series.md`, `PLAN.md` (канон). Тестирование/подтверждение GATE — отдельный
> промпт `NEXT-TESTING.md`.

## 0. Как работать с этим планом

- **Один этап = одна сессия.** Этап завершается локальным коммитом; при блокере — фиксируем в `STATE.md`,
  НЕ перескакиваем на другой этап.
- **Общий шаблон стартового промпта новой сессии** (подставить Sxx и название):
  > «Работаем над этапом **Sxx: <название>** из `NEXT-FEATURES.md` §Sxx. Прочитай сначала `STATE.md`,
  > `docs/notes/tp-artifacts.md`, `NEXT-FEATURES.md` §Sxx и `PLAN.md` (канон, не редактировать).
  > Соблюдай жёсткие ограничения проекта: только локально (без push/pull/PR), не запускать clang-format,
  > `PLAN.md` не редактировать, fallback при неделимости → репликация без throw, один этап за сессию.
  > Реализуй объём этапа, добавь тесты, обнови `STATE.md`.»
- **Критерий качественного этапа**: не регрессировать функциональные TP-тесты S05–S16; новое поведение
  включается только в TP-режиме; non-TP путь и форматы свойств/блобов не ломать (изменения формата блоба —
  только с повышением `TENSOR_PARALLEL_BLOB_VERSION`).
- Правило по неделимости/небезопасности остаётся: **репликация без throw**, классификация seam стабильна.

## Граф зависимостей

```
S17  In-graph коллективы (ядро под REPLICATED→шардинг)
 ├── S18  Peer-бэкенд + device-side события   (ускоряет S17 и текущие коллективы)
 └── S19  Zero-copy rank-import               (ускоряет загрузку TP-кеша)
S20  Подготовка upstream-серии (после стабилизации S17–S19; иначе — после S16)
```
Приоритет: **S17 → S18 → S19 → S20** (S18/S17 можно менять местами, если раньше доступно peer-железо).

---

## S17 — In-graph межранговые коллективы (снятие REPLICATED)

> **Входные факты от S16 hardware validation (2× Arc A770, 2026-10-18).** Подтверждено на железе:
> сборка/тесты/бенч выполнены; паритет ядра шардинга — зелёный. Ниже — то, что S17 **обязан** закрыть.
> 1. **Перф-потолок подтверждён замером**: `tp_bench.py` (indicative `decode_step_proxy`, GenAI нет) дал
>    `TP1 = 0.0512 s`, `TP2 = 0.0643 s` → `TP1/TP2 = 0.80` (TP2 медленнее). Профиль TP2:
>    `rank0=20.3 ms` + `rank1=20.2 ms` node-time, wall ≈64 ms → ≈24 ms/шаг невычислительного оверхеда.
>    Реальный Qwen почти целиком **REPLICATED** → нужен именно S17 (снятие REPLICATED) для вклада в ≥1.3×.
> 2. **Ограничение harness**: без `openvino_genai` реальные `TTFT`/`TPOT` не снимаются; `MULTI:GPU,GPU`
>    не запускается (`AUTO cumulative model doesn't support stateful model`). Финальный GATE-замер — только
>    с GenAI (`NEXT-TESTING.md`).
> 3. **Реальный TP-баг (блокирует S13/GenAI-паритет), фикс — вне test-only сессии:**
>    `TPCompiledModel::build_output_ports()` (`tp_compiled_model.cpp:90-96`) создаёт **синтетический**
>    full-shape `Result` для ALL_GATHER-выходов (в т.ч. выход head-sharded состояния) и **не переносит**
>    `port.get_names()`/friendly-name. Итог: `compiled.output(0).get_names()` пуст → `get_tensor("logits")`
>    и GenAI name-based binding падают (`GenAiStyleStatefulRoundtripMatchesSingleGpu`:
>    `Port for tensor name logits was not found`); plain single-GPU имена сохраняет. Фикс: при построении
>    синтетического порта скопировать `port.get_names()` на `result->output(0).get_tensor()` (и friendly
>    name). Добавить регресс-тест «TP output tensor names preserved for all-gathered outputs».
> 4. **Stateful-паритет не проверен**: 10 из 11 падений `*TensorParallel*` — pre-existing ограничения
>    GPU-плагина на stateful-моделях теста (rank-3 `ReadValue`+`Add` при «средней» оси >2 →
>    `Argument shapes are inconsistent`; rank-3 stateful depthwise `GroupConvolution`; `query_state()` batch
>    `0`). Воспроизведено на stock upstream `2026.5.0-22967` **без TP**. Нужно поправить тест-модели
>    (иные формы / явный init) либо плагин — завести отдельной задачей (не TP-регресс).

**Проблема.** Сейчас сборка результата и агрегация состояния живут на границе модели (`TPInferRequest`:
`assemble_outputs`/`assemble_reduce_sum`/`TPAggregatedState`), а промежуточные швы между слоями не
поддерживаются → реальные Qwen-слои (coupled fused `in_proj`, `Concat`/`Loop`/`ScatterNDUpdate`) остаются
**REPLICATED** → `TPOT(TP1)/TPOT(TP2) ≈ 1.0`. Нужны коллективы **между узлами графа** (all-gather после
N-split-проекции; all-reduce после K-split), чтобы шардить реальные слои.

**Объём (in):**
- Вставить коллективные узлы в трансформированный граф на швах и исполнить их на рангах.
- Два варианта реализации (решение зафиксировать в STATE):
  - **(A) Host-mediated на границах слоёв** — разбить граф на сегменты, между сегментами коллектив через
    существующий `TPTransport`; проще, но добавляет host-sync.
  - **(B) Custom TP-op** — новый internal-оп (напр. `TPAllGather`/`TPAllReduce`) с plugin-реализацией,
    исполняемый device-side; сложнее, но без лишних остановок. **Рекомендуется начать с (A)** как
    корректного MVP, затем (B).
- Первый срез: цепочка **N-split → (all-gather) → следующий GEMM** (напр. MLP `gate/up → down`) и
  **K-split → (all-reduce) → residual/next**. Один-два паттерна, детерминированно.
- Снятие ограничения `REPLICATED` ровно для поддержанных паттернов; остальное — как раньше.

**Объём (out):** device-side peer/события (это S18); zero-copy (S19); новые типы коллективов вне
all-gather/all-reduce.

**Точки кода:** `src/plugins/intel_gpu/src/plugin/transformations/tp_shard_transform.{hpp,cpp}` (вставка
коллектив-узлов, новые паттерны), при (B) — `src/plugins/intel_gpu/src/plugin/ops/*` +
регистрация implementation, `tp_infer_request.cpp`/`tp_transport.{hpp,cpp}` (исполнение коллектива),
`tp_plan.hpp` (расширить `TPOutputSeam`/новый тип seam для межслойных швов, `TENSOR_PARALLEL_BLOB_VERSION`
при изменении формата), `state-inventory.md` (обновить статус REPLICATED-слоёв).

**Инварианты:** fp32-аккумуляция (§7.4); реплицированные термы один раз; фьюжны не пересекают швы;
детерминизм порядка коллективов; fallback → репликация без throw.

**Тесты:** новые билдеры-модели с двумя+ GEMM подряд (N→GEMM и K→residual) → паритет vs 1-GPU;
проверка, что `tp_seams` показывает шардинг (не REPLICA) для реального паттерна; негатив на неделимость.

**GATE:** поддержанный паттерн реального слоя больше не REPLICATED; паритет сохранён; замер
`TPOT`/`TTFT` (см. `NEXT-TESTING.md`) показывает улучшение относительно S16 (цель — вклад в ≥1.3×/1.5×).

**Риски:** корректность числового инварианта K-split при межслойном all-reduce; производительность
host-mediated (A) при decode; сложность детерминизма при произвольном порядке коллективов.

**Артефакты:** обновлённые `tp_shard_transform.*`, `tp_plan.hpp`, `tp_transport.*`, тесты, `STATE.md`,
при (B) — новый op-файл(ы).

---

## S18 — Peer-бэкенд и device-side события

**Проблема.** На целевой платформе (2× A770) `zeDeviceCanAccessPeer` = false → peer-ветка `TPTransport`
инертна; device-side события не реализованы (S12). На железе с peer-доступом (напр. 2× BMG) можно
исключить host-round-trip и host-sync floor — главный рычаг к `TPOT ≥ 1.5×`.

**Объём (in):**
- Зарегистрировать реальный peer-бэкенд: `TPTransport::set_peer_region_copy` для прямых device-копий
  (L0 `zeCommandListAppendMemoryCopy` между устройствами/через peer-доступный буфер).
- Подключить бэкенд там, где peer доступен (`gpu_devices_peer_accessible` = true); host-staging — fallback.
- Device-side события/синхронизация для async-исполнения рангов (без host-sync на каждый коллектив);
  учесть находку S02 (counter-based event + `zeEventHostSignal` = UNSUPPORTED; классический `zeEventCreate`
  из ctypes крашил драйвер — использовать штатный путь движка, осторожно с `zeEventPool`).
- Замерить peer vs host-staging.

**Объём (out):** изменение алгоритмов коллективов (ring/all-gather уже есть); zero-copy (S19).

**Точки кода:** `src/plugins/intel_gpu/src/runtime/tp_peer.cpp`,
`src/plugins/intel_gpu/src/plugin/tp/tp_transport.{hpp,cpp}` (PeerRegionCopy-бэкенд, события),
возможно `tp_infer_request.cpp` (async-синхронизация рангов), `ze_engine`/event-обёртки движка.

**Инварианты:** на платформах без peer — поведение и результаты не меняются (host-staging, G0);
fp32; детерминизм.

**Тесты:** юнит-тест бэкенда peer-копий (если есть peer-железо/эмуляция); функциональные TP-паритеты
на peer-железе; сравнение `last_all_reduce_stats()`.

**GATE:** при peer-доступе объём host-трафика/латентности коллективов снижен и `TPOT` улучшен;
без peer — регрессий нет.

**Риски:** стабильность драйвера при событиях (см. находки S02); переносимость L0/OpenCL;
корректность синхронизации.

**Артефакты:** `tp_transport.*`, `tp_peer.cpp`, тесты, замеры в `STATE.md`.

---

## S19 — Zero-copy rank-import (mmap пер-rank блоба)

**Проблема.** Сейчас `import_model_tensor_parallel` пересобирает каждый rank-блоб из памяти
(istringstream) без mmap/zero-copy → медленная загрузка TP-кеша и лишняя память. Обычный путь имеет
zero-copy (tensor-based import + `g_current_tensor_base`).

**Объём (in):**
- Импортировать rank-блобы **по смещению** из mmap/tensor-буфера конверта (не копируя в stringstream),
  подавая в `import_single_device_model` корректный `tensor_base` (указатель на начало блоба ранга) и
  учитывая выравнивание страницы (`enable_zero_copy_cache_load`, padding как в `CompiledModel::export_model`).
- Расширить `import_single_device_model`/`BinaryInputBuffer`-использование для под-потока с базой тензора.
- Гарантировать время жизни буфера (mmap/tensor) на время жизни compiled model.

**Объём (out):** изменение формата конверта, если возможно сохранить обратную совместимость
(иначе — bump `TENSOR_PARALLEL_BLOB_VERSION`); peer/события (S18).

**Точки кода:** `src/plugins/intel_gpu/src/plugin/plugin.cpp` (`import_model_tensor_parallel`), возможно
минимальные правки `import_single_device_model` (передача tensor_base на регион), `tp_compiled_model.cpp`
(выравнивание per-rank блобов при export), `core_impl` mmap-путь (только чтение/понимание).

**Инварианты:** корректность не зависит от backend’а чтения (stream vs mmap) — одинаковые результаты;
non-TP путь не меняется.

**Тесты:** S16 roundtrip остаётся зелёным и на mmap-пути; проверка, что ранг-блобы не копируются
(инструментально/по памяти).

**GATE:** загрузка TP-кеша из mmap работает и не копирует ранги; паритет сохранён.

**Риски:** выравнивание/смещения в конверте; срок жизни mmap; отличия платформ.

**Артефакты:** `plugin.cpp`, `tp_compiled_model.cpp`, тесты, `STATE.md`.

---

## S20 — Подготовка upstream-серии PR

**Проблема/цель.** Все изменения локальны (`feature/gpu-tensor-parallel`). Нужно подготовить к выносу
в upstream по серии независимо ревьюируемых PR (PLAN §8; черновик — `docs/notes/upstream-series.md`).

**Объём (in):**
- Актуализировать `docs/notes/upstream-series.md` под фактическое состояние кода (после S17–S19):
  слайсы (PR 1→N), заголовки/объём/зависимости/тесты на каждый.
- Подготовить **локальные** ветки/стеки по PR (без push!) или упорядоченный набор патчей; убедиться,
  что каждый слайс собирается независимо и проходит свои TP-тесты.
- Чек-лист качества PR: non-TP поведение неизменно; описания; связь с PLAN; ссылки на доки.

**Объём (out):** сам push/создание PR — **запрещено** политикой (только подготовка).

**Точки кода:** все изменённые TP-файлы (см. `docs/notes/tp-artifacts.md` §2); в основном — организация
коммитов/патчей и документации.

**GATE:** план-документ актуален; для каждого слайса указаны файлы/тесты/зависимости; подготовлены
локальные патчи (при желании). Push/PR не выполняются.

**Артефакты:** `docs/notes/upstream-series.md` (обновлённый), при необходимости — локальные ветки/патчи.

---

## Сквозные жёсткие ограничения (для всех этапов)

- **Только локально: remote (origin/upstream) НЕ трогаем** — без push/pull/PR; `git submodule update`
  только при явном разрешении на сеть.
- `PLAN.md` не редактировать; дельты — в `STATE.md`.
- Не коммитить `.bin`/модельные файлы; `.bin` (4.6 GB) целиком не читать.
- **Не запускать clang-format** (репо пинует 18, в среде может быть 20+) — стиль вручную.
- Fallback при неделимости/небезопасном потребителе → **репликация без throw** (как S07–S16).
- Не регрессировать TP-тесты S05–S16; каждое изменение — только в TP-режиме.
- Версионирование блоба: изменение формата конверта → `TENSOR_PARALLEL_BLOB_VERSION++` и запись в STATE.

## Как выбрать этап для следующей сессии
1. Если доступно **peer-железо** (2× BMG, `zeDeviceCanAccessPeer`=true) — можно начать с **S18**.
2. Иначе — **S17** (максимальный вклад в перф, не зависит от железа).
3. **S19** — после S17/S18 (или сразу, если приоритет — скорость загрузки).
4. **S20** — последним, когда S17–S19 стабилизированы.
