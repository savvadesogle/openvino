# СЕССИЯ S06 — Stage 2.1 TPShardTransform-скелет: детект MatMul/FC + TPPlan · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §4.2 (compile-time,
     место TP-пасса и срез констант), §4.3 (шардирование классов слоёв), §5 (строка S06),
     §7 (риски: фьюжны через шов, срез до `propagate_constants`).
  2) C:\openvino-tp\STATE.md — живое состояние (особенно «Заметки сессия → следующая»: разбор
     артефактов S04/S05 — `TPCoordinator`, `TPCompiledModel`, `TPInferRequest`, `TPPlan`; «Блокеры» —
     сборка недоступна).
  3) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3 (OV-пассы и `propagate_constants`):
     где именно вставлять пасс и почему срез весов обязан быть до upload.
  4) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — §3–§6 (N-split FC только у CPU; K-split/allreduce
     проектируется с нуля; численные инварианты int4/K кратно 128).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S06 (таблица PLAN.md §5).
  Следующие этапы (S07+ — реальный N-split/K-split/коллективы) запрещены. Остановка на GATE или
  блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; `docs/notes/`; L0-спайки (2× Arc A770,
  peer/IPC NA). **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e).
- S04 done (частично): Stage 1.1 `TPCoordinator` — TP-ветка `Plugin::compile_model`:
  - `Plugin::is_tensor_parallel_requested(config)` (policy{TENSOR_PARALLEL} ИЛИ internal flag) и
    `Plugin::compile_model_tensor_parallel(...)` — `src/plugins/intel_gpu/src/plugin/tp/tp_coordinator.cpp`;
    перечисляет все GPU из `m_device_map`, по `RemoteContextImpl`+`ExecutionConfig` на ранг, клонирует
    модель per-rank (`clone_and_transform_model` + `finalize` + `CompiledModel`), оборачивает в `TPCompiledModel`.
  - `TPCoordinator`/`TPPlan` (`include/intel_gpu/plugin/tp/tp_{coordinator,plan}.hpp`). **`TPPlan.seams`
    сейчас ПУСТ, `TPSeamKind` = только `NONE`** — заполнение начинается в S06.
  - Свойства: `ov::intel_gpu::tp_ranks` (RO `vector<int32_t>`), `ov::intel_gpu::enable_tensor_parallel`
    (RW bool) — `runtime/internal_properties.hpp`; опции `ov::hint::model_distribution_policy`,
    `ov::intel_gpu::enable_tensor_parallel` — `runtime/options.inl`.
- S05 done (частично): Stage 1.2 `TPInferRequest` — **ранги идентичны (нет шардинга)**:
  - `TPInferRequest` (`include/.../tp/tp_infer_request.hpp`, `src/.../tp/tp_infer_request.cpp`): k per-rank
    sub-запросов (`get_rank_models()`), репликация входов (общий host-тензор в `set_tensor` + memcpy
    rank0→rankN в `infer()` для lazy), выходы только rank0, последовательный `rank->infer()`.
  - `TPCompiledModel::create_sync_infer_request()` → `TPInferRequest`; `create_infer_request()` →
    `create_async_infer_request<>()` (в ctor — свой `CPUStreamsExecutor` через `set_task_executor`).
  - GATE-тест: `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`
    (`TensorParallelInferMatchesSingleGpu`, `make_tp_parity_model`).
- **Среда сборки НЕДОСТУПНА**: тулчейн есть (cl 14.44/cmake/ninja), но **все git-сабмодули пусты**
  (`git submodule status` → все с `-`), cmake configure падает; инициализация = сетевой fetch (запрещён).
  См. STATE «Блокеры». Если сабмодули появились — собери и прогони `tensor_parallel` тест; иначе честный
  частичный GATE.
- Тест-модель `models/Qwen3.5-9B-int4-ov/` — в S06 желательна для «детект полон и стабилен» (LM-граф),
  но мини-сеть (FC/MatMul) достаточна для позитива; полный Qwen — S07+.

## Задача S06 (полный объём этой сессии)
Stage 2.1 — **TPShardTransform-скелет**: разметка швов БЕЗ изменения численного результата. Пасс
детектирует кандидаты на шардирование (`MatMul`/`FullyConnected`), классифицирует их тип (для будущих
N-split / K-split / реплики), и заполняет `TPPlan.seams`. Никакого фактического среза весов и никаких
коллективов на этом этапе — граф/результат должны остаться идентичными (identity-пасс).

Работать в `src/plugins/intel_gpu/**`. НЕ реализовывать S07+ (реальный N-split / all-gather), S08+
(K-split / allreduce), коллективы, per-rank KV. GATE — модель компилируется с пассом, паритет
(1-GPU vs TP-реплика) сохраняется, а `TPPlan` содержит стабильный, полный список швов.

1. **Новый OV-пасс `TPShardTransform`** (identity/analysis):
   - Разместить рядом с прочими плагинными пассами `TransformationsPipeline` (см.
     `plugin/transformations_pipeline.cpp`, старт `::apply` ~`:555`), **до фьюжнов и LowPrecision**
     (ориентиры: SDPAToVLSDPA, TransposeMatMul, MarkDequantization, MoE-группа, GatedDeltaNetFusion),
     и семантически ДО `propagate_constants` (см. gpu-plugin-arch.md §3: `propagate_constants`
     материализует константы на устройстве — срез обязан быть раньше; сейчас среза нет, но архитектурно
     пасс ставим в правильную точку).
   - Пасс выполняется ТОЛЬКО в TP-режиме. Прокинуть признак TP в `TransformationsPipeline`
     (напр. параметр/флаг из `ExecutionConfig::get_enable_tensor_parallel()`/policy), чтобы при
     обычной компиляции граф не менялся вовсе (регресс-инвариант: TP-off == upstream).
   - На S06 пасс — **analysis-only**: он НЕ заменяет/не добавляет ноды, только собирает метаданные
     (`rt_info`-маркировка или отдельная структура) о найденных швах. Доказательство identity:
     после пасса `model` изоморфна входной (численно и структурно; можно ассертить кол-во нод).
   - Механизм пересылки результата пасса в ранги: пасс наполняет данные, которые далее агрегируются в
     `TPPlan.seams` (см. п.2). Продумать, как метаданные доезжают из `TransformationsPipeline` в
     `TPCoordinator`/`TPCompiledModel` (варианты: `rt_info` модели + сбор после `clone_and_transform_model`
     в `compile_model_tensor_parallel`; или out-параметр пайплайна). Выбрать минимально инвазивный путь
     и зафиксировать в STATE.

2. **Расширить `TPPlan`/`TPSeamKind`** (`include/intel_gpu/plugin/tp/tp_plan.hpp`):
   - Добавить виды швов, соответствующие PLAN §4.3 (напр. `N_SPLIT`, `K_SPLIT`, `REPLICA`; exactly
     то, что реально детектится на этом этапе; ввод «реплика» допустим, если полезен).
   - `TPSeam` дополнить полями, которые понадобятся S07/S08 (имя ноды/тензора, тип, ось/группа,
     режим коллектива), но БЕЗ реализации поведения. Документировать, что поля пока не исполняются.
   - `TPCoordinator::make_plan()` заполняет `seams` из метаданных пасса (детерминированный порядок;
     стабильность списка — требование GATE: одинаковый вход → одинаковый план).

3. **Классификация** (по PLAN §4.3; на этом этапе — только детект/тип, без среза):
   - `MatMul`/`FullyConnected` → определить, какой тип шва к нему применим в будущем (N-split по
     out-channels или K-split + allreduce). Правило классификации можно начать консервативно
     (например, первый MatMul слоя → N-split, out-proj → K-split), но обязательно **детерминированно**
     и покрыто тестом. При неопределённости — безопасный дефолт (`REPLICA`/`NONE`) + лог.
   - Прочие узлы (LayerNorm/RMS, RoPE, активации) — пока не трогать (реплики по умолчанию).

4. **Тесты/GATE** (расширить `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`
   или соседний файл того же каталога — ROOT-glob подхватывает без правки CMake):
   - Позитив детекта: мини-граф с несколькими `MatMul`/`FC` → TP-компиляция успешна; `TPPlan`
     (доступен через `get_tp_plan()`/`tp_ranks`-инфраструктуру S04) содержит непустой, стабильный,
     ожидаемый список швов. Если план не виден через публичный API — добавить минимальный тестовый
     доступ (например, RO property `ov::intel_gpu::tp_plan`/`tp_seams`), задокументировав.
   - **Identity-инвариант**: паритет 1-GPU vs TP (как в S05 `TensorParallelInferMatchesSingleGpu`)
     сохраняется — пасс не меняет числа. SKIP при <2 GPU.
   - Регресс TP-off: обычная компиляция (без policy) не запускает пасс / план пуст
     (`tp_ranks` == {}).
   - Не ломать негативы S04 (`"GPU.0"`+TP, `PIPELINE_PARALLEL`).
   - Для «детект полон и стабилен» на Qwen3.5 LM (PLAN §5 GATE) — если модель под рукой и сборка
     доступна: собрать LM-граф, прогнать пасс, проверить, что число/типы швов соответствуют
     структуре (24 linear_attention + 8 full_attention; GQA 16q/4kv; hidden 4096; MLP 12288).
     Иначе — отложить как частичный GATE с явной пометкой.

5. **Регистрация**: новые `.cpp` под `src/plugin/**` подхватываются `file(GLOB_RECURSE PLUGIN_SOURCES
   src/plugin/*.cpp ...)` в `src/plugins/intel_gpu/CMakeLists.txt` (re-run cmake). Тест — ROOT-glob.
   Убедиться, что включение пасса не требует правок CMake (при необходимости — минимальные).

## GATE S06 (критерии завершения)
- `TPShardTransform` (identity) встроен в `TransformationsPipeline`, активен только в TP-режиме;
  TP-off граф не изменяется.
- `TPPlan.seams` наполняется детектированными швами (`MatMul`/`FC`), список стабилен и полон для
  целевого графа; `TPSeamKind` расширен под S07/S08.
- Модель компилируется с пассом; паритет S05 (1-GPU vs TP2) сохранён (по плану/коду; при недоступной
  сборке — частично, с явной пометкой).
- STATE.md: S06 done (или честный блокер) + артефакты + решения + заметки → S07.
- NEXT.md — полный самодостаточный промпт S07 (Stage 2.2 N-split + all-gather-эквивалент: срез
  констант весов по out-channels ДО upload, сборка полного выходного буфера без арифметики;
  GATE — FC/MLP сеть паритет cos>0.999).
- Commit (сообщение: "S06: TPShardTransform skeleton (seam detection, identity)") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S07+ (реальный срез весов / all-gather / K-split / allreduce),
  коллективы, per-rank KV. S06 — только разметка швов, численный результат не меняется.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если нужна сборка OpenVINO и она недоступна (сабмодули пусты) — не изобретать обходные сборки;
  зафиксировать блокер в STATE и считать GATE частичным (код + план теста готовы).
