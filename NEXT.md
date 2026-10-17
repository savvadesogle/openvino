# СЕССИЯ S05 — Stage 1.2 TPInferRequest: реплика графа на 2 GPU, выходы rank0 · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §3 (ключевые
     факты кода), §4.3–4.4 (дизайн infer/обёртки), §5 (строка S05), §7 (риски).
  2) C:\openvino-tp\STATE.md — живое состояние (особенно «Заметки сессия → следующая» — там
     полный разбор артефактов S04 и что именно осталось заглушкой; «Блокеры» — про сборку).
  3) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — архитектура GPU-плагина (infer request,
     `compiled_model.cpp`, `async_infer_request.*`, `remote_context.cpp`).
  4) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — как CPU оформляет TP-потоки/свойства.
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S05 (таблица PLAN.md §5).
  Следующие этапы (S06+) запрещены, даже если остаётся время/контекст. Остановка на GATE
  или блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; конспекты в `docs/notes/`; L0-спайки
  (2× Arc A770, peer/IPC NA). **Gate G0 РЕШЁН: дефолт коллективов = host-USM staging**
  (dev→pinned host→dev, ~4–5 GB/s e2e). В S05 коллективы всё ещё **НЕ** строятся.
- **S04 done (частично, 2026-10-17)** — Stage 1.1 `TPCoordinator` (TP-ветка compile + свойства
  + каркас обёртки). Уже реализовано и, по задумке, будет переиспользовано:
  - `Plugin::is_tensor_parallel_requested(config)` и `Plugin::compile_model_tensor_parallel(...)`
    (`src/plugins/intel_gpu/src/plugin/tp/tp_coordinator.cpp`): перечисляют все GPU из
    `m_device_map`, делают по одному `RemoteContextImpl`+`ExecutionConfig` на ранг, клонируют
    модель per-rank (`clone_and_transform_model` + `finalize` + `CompiledModel`), затем
    оборачивают в `TPCompiledModel`.
  - `TPCoordinator`, `TPPlan` (`include/intel_gpu/plugin/tp/tp_{coordinator,plan}.hpp`).
  - `TPCompiledModel` (`include/.../tp_compiled_model.hpp`, `src/.../tp/tp_compiled_model.cpp`):
    **СЕЙЧАС `create_infer_request()`/`create_sync_infer_request()` делегируют в rank-0** —
    ЭТО ГЛАВНАЯ ЗАГЛУШКА, которую S05 должен заменить. Обёртка уже отдаёт `ov::intel_gpu::tp_ranks`
    и `get_rank_models()`/`get_tp_plan()`.
  - Свойства: `ov::intel_gpu::tp_ranks` (RO, `std::vector<int32_t>`) и
    `ov::intel_gpu::enable_tensor_parallel` (RW, bool) в `runtime/internal_properties.hpp`;
    опции `ov::hint::model_distribution_policy` и `ov::intel_gpu::enable_tensor_parallel` в
    `runtime/options.inl`. В `compiled_model.cpp` `get_property(tp_ranks)` → `{}` для не-TP.
  - Тест S04: `tests/functional/behavior/tensor_parallel.cpp` (компиляция 2-GPU мини-FC,
    `tp_ranks=={0,1}`, негативы `"GPU.0"`+TP и `PIPELINE_PARALLEL`).
- **Среда сборки НЕДОСТУПНА**: тулчейн есть (cl 14.44/cmake/ninja), но **все git-сабмодули
  пусты** (gtest, ittapi, json, ocl, onednn_gpu и т.д.), `cmake` configure падает; инициализация
  сабмодулей = сетевой fetch (запрещён). См. STATE «Блокеры». Если сабмодули вдруг появились —
  выполни минимальную проверку компиляции целевых файлов; иначе зафиксируй честный частичный GATE.
- Тест-модель: `models/Qwen3.5-9B-int4-ov/` — в S05 НЕ обязательна, достаточно мини-сети.

## Задача S05 (полный объём этой сессии)
Stage 1.2 — **TPInferRequest**: корректное исполнение TP-реплики модели (без шардирования:
ранги идентичны) на k GPU с выдачей выходов ранга 0. Цель GATE — **e2e паритет**: TP-инференс
2-GPU мини-сети даёт тот же результат, что одиночный (1-GPU) инференс той же модели.

Работать в `src/plugins/intel_gpu/**`. Не реализовывать S06+ (TPShardTransform, шардирование),
транспорт коллективов (S08+), per-rank KV (S10). Ранги на этом этапе идентичны.

1. **TPInferRequest** (PLAN §4.3–4.4):
   - Заменить делегирование в rank-0 внутри `TPCompiledModel`. Создать
     `TPInferRequest`/`TPSyncInferRequest` (по образцу `ISyncInferRequest`/существующего
     `AsyncInferRequest`): держит k под-инференс-реквестов (по одному на ранговый
     `CompiledModel` из `get_rank_models()`).
   - Inputs: реплицировать входные тензоры **в каждый ранг** (memcpy входного тензора в
     `set_tensor` каждого rank-реквеста). Outputs: отдавать тензоры **только ранга 0**.
     Согласовать `get_tensor`/`set_tensor`/`get_tensors`/`set_tensors` со свойствами
     `IInferRequest` (dev API `src/inference/dev_api/openvino/runtime/isync_infer_request.hpp`).
   - Sync-инференс: последовательно (MVP) `rank->infer()` по всем рангам. Async — обернуть
     sync в `ov::IAsyncInferRequest` (как делает существующий плагин) через
     `create_async_infer_request<...>()` или явный класс.
   - В `TPCompiledModel::create_sync_infer_request()`/`create_infer_request()` вернуть новый
     TPInferRequest вместо rank-0.
2. **Inputs/outputs обёртки**: убедиться, что `TPCompiledModel::inputs()/outputs()` берутся
   из ранга 0 (уже так) и совпадают с ранговыми графами (ранги идентичны).
3. **Тест/GATE** (расширить существующий `tests/functional/behavior/tensor_parallel.cpp`
   или добавить соседний файл того же каталога — ROOT-glob подхватит без правки CMake):
   - Мини-модель (Parameter → MatMul/FC → Result) со **предсказуемыми** весами.
   - Reference: `core.compile_model(model, "GPU")` (1 GPU) → infer.
   - TP: `core.compile_model(model, "GPU", {model_distribution_policy: TENSOR_PARALLEL})` → infer.
   - `ASSERT` близкое равенство выходов (e.g. `ov::test::utils::compare` или поэлементно,
     допуск ~1e-4). SKIP при <2 GPU.
   - Сохранить/не сломать негативные кейсы S04 (`"GPU.0"`+TP, `PIPELINE_PARALLEL`).
4. **Регистрация**: проверить, что новые `.cpp` в `src/plugin/tp/**` подхватываются
   `file(GLOB_RECURSE ...)` в `CMakeLists.txt` плагина (нужен re-run cmake); тест — ROOT-glob.

## GATE S05 (критерии завершения)
- `TPInferRequest` создаётся, исполняет k рангов, входы реплицируются, выходы — ранга 0.
- e2e паритет TP (2 GPU) vs 1-GPU на мини-сети (по плану/коду; при недоступной сборке —
  частично, с явной пометкой).
- STATE.md: S05 done (или честный блокер) + артефакты + решения + заметки → S06.
- NEXT.md — полный самодостаточный промпт S06 (Stage 2.1 TPShardTransform-скелет:
  разметка швов без изменения численного результата; GATE — модель компилируется с
  identity-пассом, паритет сохраняется).
- Commit (сообщение: "S05: TPInferRequest (replicated graph, rank0 outputs)") — **локально,
  без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений
  удалённого репозитория, включая `git submodule update` (это сетевой fetch).
- Атомарность этапа: не реализовывать S06+ (шардирование/TPShardTransform), коллективы,
  per-rank KV. Ранги идентичны, коллективы не нужны.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если нужна сборка OpenVINO и она недоступна (сабмодули пусты) — не изобретать обходные
  сборки; зафиксировать блокер в STATE и считать GATE частичным (код + план теста готовы).
