# STATE.md [копия для переноса в C:\openvino-tp\STATE.md]
Канон — PLAN.md (не редактировать в сессии). Здесь: прогресс, решения, блокеры.

## Управление
| Поле | Значение |
|---|---|
| Ветка | feature/gpu-tensor-parallel (активна, **только локально**) |
| Remote-политика | **Работаем только локально, remote НЕ трогаем** (решение 2026-10-17): коммиты не пушим, origin/upstream не изменяем |
| Upstream tip базис | 85720fe2 (2026-10-17) |
| ОС / железо | Windows primary; 2+ Intel Arc/BMG |
| Тест-модель | models/Qwen3.5-9B-int4-ov (полная, кроме openvino_mtp_model.*) |
| Коллективы дефолт | **host-USM staging** (pinned `zeMemAllocHost` + per-device command queue; G0 решён в S03). Peer/IPC/кросс-shared USM на целевой машине недоступны; peer — runtime fast-path |

## Прогресс сессий
- [x] Планирование (2026-10-17, PLAN.md готов)
- [x] Базовая инициализация репо (2026-10-17): remotes настроены (origin/upstream); shallow-fetch tip 85720fe2; commit 6339e518 (PLAN/STATE/NEXT/.gitignore). Далее — работа только локально.
- [x] S01 init (2026-10-17): ветка feature/gpu-tensor-parallel от master; docs/notes/{cpu-tp-pattern,gpu-plugin-arch,hetero-pp,l0-spike-plan}.md; STATE обновлён; NEXT — промпт S02; commit 7bd6c6d4 (локально)
- [x] S02 L0-спайк A (2026-10-17): tools/l0_spike/{l0_capability.py,README.md,l0_capability_output.txt}; результаты — docs/notes/l0-spike.md; capability-матрица зафиксирована (GATE пройден)
- [x] S03 L0-спайк B (G0) (2026-10-17): tools/l0_spike/l0_bench.py (+ output); bw/latency-таблицы — docs/notes/l0-spike.md §8; **G0 = host-USM staging** (журнал гейтов)
- [x] S04 Stage1.1 TPCoordinator (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S05 Stage1.2 TPInferRequest (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S06 Stage2.1 TPShardTransform-скелет (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [ ] S07 Stage2.2 N-split + all-gather
- [ ] S08 Stage2.3 K-split + allreduce
- [ ] S09 Stage2.4 int4_ASYM
- [ ] S10 Stage3.1 per-rank KV
- [ ] S11 Stage3.2 linear_attention
- [ ] S12 Stage3.3 Hard (dynamic/events/concurrency)
- [ ] S13 Stage4.1 GenAI интеграция
- [ ] S14 Stage4.2 Scheduler + бенч
- [ ] S15 Stage5.1 перф
- [ ] S16 Stage5.2 стабилизация

## Гейт-решения (журнал)
| Дата | Гейт | Решение | Обоснование |
|---|---|---|---|
| 2026-10-17 | S01 init | done (локально) | артефакты GATE готовы: ветка активна, 4 конспекта в docs/notes, STATE/NEXT обновлены; commit 7bd6c6d4 локально |
| 2026-10-17 | S02 L0-спайк A | done | capability-матрица известна и записана в docs/notes/l0-spike.md: 2×A770; canAccessPeer кросс = false; P2P кросс = 0x0; IPC-память UNSUPPORTED_FEATURE (core и ZEX_mem_ipc_handles); counter-based IPC event GetIpcHandle = OUT_OF_DEVICE_MEMORY. Решение G0 — в S03 (перф не мерялась) |
| 2026-10-17 | **S03 L0-спайк B → G0** | **G0 = host-USM staging** | Микробенч (docs/notes/l0-spike.md §8): единственный рабочий междевайсный путь — dev_i→pinned host→dev_j; ~7–10 GB/s/плечо, ~4–5 GB/s end-to-end (16–128 MB); локальная same-device полоса 145–192 GB/s; кросс-shared USM = OUT_OF_DEVICE_MEMORY; peer/IPC NA (S02). Дефолт коллективов TP — host-USM staging; peer — runtime fast-path по `zeDeviceCanAccessPeer` |
| 2026-10-17 | S04 Stage1.1 TPCoordinator | **частично** (код+тест готовы) | TP-ветка `compile_model`, `TPCoordinator`, `TPCompiledModel`, свойства (`model_distribution_policy`/`enable_tensor_parallel`/`tp_ranks`), GATE-тест `tensor_parallel.cpp` написаны. **GATE по сборке не подтверждён**: тулчейн (cl 14.44/cmake/ninja) есть, но ВСЕ сабмодули пусты (`git submodule status` → все с `-`; напр. `thirdparty/ittapi/ittapi`, `gtest`, `json/nlohmann_json`, `ocl/*`, `intel_gpu/thirdparty/onednn_gpu` = 0 файлов) → cmake configure падает (`ittapi::ittnotify does not represent a cmake target`). Инициализация сабмодулей = сетевой fetch → запрещён политикой. См. «Блокеры» |
| 2026-10-17 | S05 Stage1.2 TPInferRequest | **частично** (код+тест готовы) | Заглушка rank-0 заменена на `TPInferRequest` (`include/.../tp/tp_infer_request.hpp`, `src/.../tp/tp_infer_request.cpp`): k per-rank sub-запросов, репликация входов (общий host-тензор в `set_tensor` + memcpy rank0→rankN в `infer()` для lazy-буферов), выходы только rank0; `infer()` — последовательно по рангам (MVP). `TPCompiledModel::create_infer_request()` → `create_async_infer_request<>()` (обёртка `ov::IAsyncInferRequest` на выделенном `CPUStreamsExecutor`), `create_sync_infer_request()` → `TPInferRequest`. GATE-тест `TensorParallelInferMatchesSingleGpu` (паритет TP2 vs 1-GPU, доп. модель `make_tp_parity_model` с различными весами, SKIP <2 GPU). **GATE по сборке/прогону не подтверждён** — сабмодули по-прежнему пусты (см. S04/«Блокеры»)
| 2026-10-17 | S06 Stage2.1 TPShardTransform | **частично** (код+тест готовы) | Добавлен identity/analysis OV-пасс `TPShardTransform` (`src/plugin/transformations/tp_shard_transform.{hpp,cpp}`): детектит `MatMul`/`FullyConnected`(`Compressed`), классифицирует (N_SPLIT/K_SPLIT/REPLICA) детерминированно, пишет `std::vector<TPSeam>` в `model->get_rt_info()` под ключом `intel_gpu_tp_seams`. Пасс регистрируется в `TransformationsPipeline::apply` **только при TP** (policy{TENSOR_PARALLEL} ИЛИ `enable_tensor_parallel`) — до фьюжнов/LowPrecision и семантически до `propagate_constants`. `TPPlan`/`TPSeamKind`/`TPCollectiveKind` расширены (`tp_plan.hpp` + новый `tp_plan.cpp`), `TPCoordinator::compile_model_tensor_parallel` собирает seams ранга-0 в план. Новое RO-свойство `ov::intel_gpu::tp_seams` (`vector<string>`, формат `KIND:COLLECTIVE:TYPE:NAME`) на `TPCompiledModel` и пустое на обычном/плагинном. Тесты S06 в `tensor_parallel.cpp` (детект+стабильность, K_SPLIT-классификация residual-проекции, TP-off регресс). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры»); identity подтверждается контрактом пасса (return false, только rt_info) |

## Блокеры / открытые вопросы
- **Сборка OpenVINO невозможна в этой среде (2026-10-17, S04).** Тулчейн на месте:
  MSVC `cl` 14.44 (VS 2022 Community), `cmake` 4.0.3, `ninja` — но все git-сабмодули
  не инициализированы (пустые каталоги). `cmake -G Ninja -S . -B build_tp ...` падает на
  `thirdparty/ittapi` (`ittapi::ittnotify does not represent a cmake target`), а также
  отсутствуют `gtest`, `json/nlohmann_json`, `ocl/{cl_headers,clhpp_headers,icd_loader}`,
  `flatbuffers`, `intel_gpu/thirdparty/onednn_gpu` и др. `git submodule update --init`
  = сетевой fetch (запрещён remote-политикой) и многогигабайтная закачка. Итог: S04 GATE
  подтверждён только статически (ревью кода/сигнатур); сборка/запуск теста — вне сессии.
  Актуально для S05 и S06: GATE подтверждён по коду/плану, прогон `tensor_parallel` тестов — вне сессии.
- **Решено (2026-10-17): remote не используем.** Сессия работает исключительно в
  локальном репозитории `C:\openvino-tp`; на `origin`/`upstream` ничего не пушим и
  не изменяем. Формулировка «push» в старых записях/промптах считается снятой.
  Локальные коммиты S01/S02: 7bd6c6d4, d0d2a20c, b97946ad, e6cbc7d5 (и далее).
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- **S06 done (частично, 2026-10-17)** — Stage 2.1 TPShardTransform-скелет. Артефакты:
  - Новый OV-пасс `TPShardTransform` (`src/plugins/intel_gpu/src/plugin/transformations/tp_shard_transform.hpp`,
    `.cpp`): `ov::pass::ModelPass`, `run_on_model` возвращает **false** (identity — граф не меняется).
    Идет по `model->get_ordered_ops()` (детерминированный порядок), для каждого `MatMul`/`FullyConnected`
    (`FullyConnectedCompressed`) создаёт `TPSeam{node_name,node_type,kind,collective}` и пишет весь
    `std::vector<TPSeam>` в `model->get_rt_info()[ov::intel_gpu::tp_seams_rt_info_key()]` (ключ
    `"intel_gpu_tp_seams"`). Никакого среза/all-gather/allreduce — только метаданные.
  - **Классификация (консервативная, детерминированная):** нет константных весов (операнд 1) → `REPLICA`;
    иначе если выход фида в `ov::op::v1::Add` (residual-проекция, выход→Add) → `K_SPLIT`
    (`ALL_REDUCE_SUM`); иначе → `N_SPLIT` (`ALL_GATHER`). Для Qwen это даёт qkv/gate/up = N_SPLIT,
    o_proj/down_proj = K_SPLIT (совпадает с PLAN §4.3). Attention-скоры (Q·Kᵀ, B не константа) = REPLICA.
  - **Точка вставки:** `TransformationsPipeline::apply` — основной блок `ov::pass::Manager("Plugin:GPU")`,
    регистрация `TPShardTransform` сразу после `set_per_pass_validation(false)` и **до** `SDPAToVLSDPA`;
    при этом `MatMul` ещё не сконвертирован в FC (`ConvertMatMulToFullyConnected` только на ~:1740).
    Гейт активности: `config.get_enable_tensor_parallel() || policy.count(TENSOR_PARALLEL)` — при обычной
    компиляции пасс НЕ регистрируется (регресс TP-off сохранён).
  - **Транспорт метаданных:** `tp_plan.hpp` (расширен: `TPSeamKind{NONE,N_SPLIT,K_SPLIT,REPLICA}`,
    `TPCollectiveKind{NONE,ALL_GATHER,ALL_REDUCE_SUM}`, поля `node_name/node_type/collective/split_axis/
    shard_dim_size` — последние два placeholder для S07/S08; `tp_seam_to_string()`); новый `tp_plan.cpp`
    (`set_tp_seams`/`get_tp_seams` через model rt_info, `to_string`). `TPCoordinator::
    compile_model_tensor_parallel` после `clone_and_transform_model` ранга-0 вызывает
    `get_tp_seams(transformed_model)` и кладёт в `plan.seams`. Выбран вариант «rt_info модели +
    сбор после clone_and_transform_model», минимум инвазивности.
  - **API для теста:** RO-свойство `ov::intel_gpu::tp_seams` (`std::vector<std::string>`,
    `"KIND:COLLECTIVE:TYPE:NAME"`) — `internal_properties.hpp`; отдаётся `TPCompiledModel`,
    пусто на обычном `CompiledModel`/плагине; в supported props везде.
  - **Тесты S06** (в `tensor_parallel.cpp`): `ShardTransformDetectsAndReportsSeams` (1 шов
    `N_SPLIT:ALL_GATHER:MatMul:*`, стабильность при повторной компиляции, наличие в supported),
    `ShardTransformClassifiesResidualProjectionAsKSplit` (MatMul→Add = `K_SPLIT:ALL_REDUCE_SUM`),
    `DefaultCompileHasNoSeams` (TP-off: пустые seams/ranks). Негативы S04 и паритет S05 не тронуты.
  - **НЕ делалось (S07+):** реальный срез весов, all-gather/allreduce, коллективы, per-rank KV;
    `split_axis`/`shard_dim_size` не заполняются. `make_plan()` по-прежнему даёт только ranks;
    seams досыпаются в compile-ветке.
  - **Заметка S06→S07:** S07 = N-split + all-gather-эквивалент. Срез констант делать в
    `TPShardTransform` (он уже стоит до `propagate_constants`, PLAN §4.2/§7.2): заменить Constant-веса
    на слайс `[.., n_start:n_end]` (N-split по out-channels, целыми oc-группами), добавить узел сборки
    (сейчас — identity; в S07 полный выходной буфер без арифметики). GATE — FC/MLP-сеть cos>0.999.
    ВНИМАНИЕ: все ранги сейчас получают ОДИН И ТОТ ЖЕ transformed model (clone per-rank), пасс должен
    получать rank-индекс (сейчас планируется через `tp_ranks`/config), иначе срез будет одинаковым.
- **S05 done (частично, 2026-10-17)** — Stage 1.2 TPInferRequest. Артефакты:
  - Новый sync-запрос `TPInferRequest` (`src/plugins/intel_gpu/include/intel_gpu/plugin/tp/tp_infer_request.hpp`,
    `src/plugins/intel_gpu/src/plugin/tp/tp_infer_request.cpp`): держит `m_rank_requests`
    (по `rank_model->create_sync_infer_request()` на каждый ранг из `TPCompiledModel::get_rank_models()`).
  - `TPCompiledModel::create_sync_infer_request()` → `TPInferRequest`; `create_infer_request()` →
    `create_async_infer_request<>()` (дефолтная `ov::IAsyncInferRequest`; в S05 добавляется свой
    `CPUStreamsExecutor` в ctor через `set_task_executor`, т.к. база создаётся с `nullptr` executor).
  - **Контракт реплики**: `set_tensor(input)` → один и тот же host-тензор ставится во ВСЕ ранговые
    запросы; `set_tensor(output)` → только rank0. `infer()` перед стартом вызывает
    `replicate_inputs()`: для каждого input-порта, если rank0- и rankN-тензоры — разные указатели
    (lazy-аллокация, каждый ранг создал свой буфер), `memcpy` rank0→rankN (device/remote тензоры
    отклоняются — `OPENVINO_ASSERT`; строки копируются поэлементно). Затем последовательный
    `rank->infer()`. `get_tensor`/`get_tensors` — из rank0; `query_state` — rank0 (агрегация per-rank
    KV/linear-attn — S10/S11); `get_profiling_info` — с префиксом `rankN:`; `check_tensors` — no-op.
  - **Осознанно НЕ делалось** (S06+): шардинг/`TPShardTransform`, коллективы, device-side швы,
    batched (`set_tensors`>1 → `NOT_IMPLEMENTED`), per-rank state.
- **GATE-тест S05**: расширен `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`:
  `make_tp_parity_model()` (MatMul 8×16 с различающимися знакопеременными весами) +
  `TensorParallelInferMatchesSingleGpu` (reference `"GPU"` vs TP `TENSOR_PARALLEL`, `ov::test::utils::compare`
  abs/rel 1e-4, SKIP при <2 GPU). Негативы S04 (`"GPU.0"`+TP, `PIPELINE_PARALLEL`) сохранены.
- **Заметка для S06**: тест/GATE S05 не прогонялся (сборка недоступна, сабмодули пусты). Перед S06
  нужно ревью кода; при появлении сабмодулей — собрать GPU-плагин + `tensor_parallel` тест, получить
  зелёный паритет (тогда S04+S05 GATE закрываются постфактум). S06 начинает `TPShardTransform`:
  identity/скелет-пасс, детект MatMul/FC и заполнение `TPPlan.seams`, без изменения численного
  результата (паритет должен сохраняться). Точка вставки пасса — `TransformationsPipeline::apply`
  (`transformations_pipeline.cpp:555+`), ДО фьюжнов и `propagate_constants` (срез весов — до upload,
  PLAN §4.2, risk §7.2).
- **S04 (для контекста)** — Stage 1.1 TPCoordinator. Артефакты:
  - Свойства: `ov::intel_gpu::tp_ranks` (RO, `std::vector<int32_t>`) и
    `ov::intel_gpu::enable_tensor_parallel` (RW, bool) —
    `include/intel_gpu/runtime/internal_properties.hpp`; опции
    `ov::hint::model_distribution_policy` (default пустой set) и
    `ov::intel_gpu::enable_tensor_parallel` (default false) — `runtime/options.inl`.
  - Новые файлы `include/intel_gpu/plugin/tp/{tp_plan,tp_coordinator,tp_compiled_model}.hpp`
    и `src/plugin/tp/{tp_coordinator,tp_compiled_model}.cpp` (каркас; шардинг/коллективы НЕ
    реализованы — S06+/S08+; ГЛАВНОЕ для S05: `TPCompiledModel::create_infer_request`
    сейчас делегирует в rank-0 — заглушка).
  - `plugin.cpp`: PP-проверка (`PIPELINE_PARALLEL`→throw), TP-ветка в `compile_model`,
    TP-отказ при explicit remote context, `tp_ranks` в `get_property` и supported props;
    `model_distribution_policy`/`enable_tensor_parallel` в supported props (RW).
  - `compiled_model.cpp`: `tp_ranks` (+ RO `model_distribution_policy`/`enable_tensor_parallel`)
    в supported props; `get_property(tp_ranks)` → `{}` для не-TP.
  - GATE-тест: `tests/functional/behavior/tensor_parallel.cpp` (позитив 2-GPU + internal flag
    + default + негативы: explicit `"GPU.0"`, `PIPELINE_PARALLEL`).
- **Решение S04**: TP-группа = все перечисленные GPU (`m_device_map`); каждый ранг получает
  СВОЙ `RemoteContextImpl`+`ExecutionConfig` (`remote_context.cpp:59` требует `devices.size()==1`),
  модель клонируется per-rank через существующий `clone_and_transform_model`+`finalize`+
  `CompiledModel` (шардинга нет → каждый ранг = полный граф). Обнаружение TP:
  `Plugin::is_tensor_parallel_requested` (policy{TENSOR_PARALLEL} ИЛИ internal flag).
  Explicit device id определяется по наличию `ov::device::id` в `orig_config` (Core кладёт
  его только для явного `"GPU.0"`).
- **ВАЖНО для S05** (Stage 1.2 TPInferRequest): `TPCompiledModel` уже создаёт k рангов и
  держит `get_rank_models()`/`get_tp_plan()`. Нужно заменить делегирование в rank-0 на
  реальный TPInferRequest: реплицировать входы на все ранги, инфер всех k графов, отдать
  выходы rank-0; GATE — e2e паритет vs 1-GPU. Коллективы (all-gather/allreduce) ещё НЕ нужны
  (шардирования нет → ранги идентичны). Среда сборки недоступна (см. «Блокеры»).
- **Вход S04 из S03 (G0)**: дефолт коллективов = **host-USM staging**
  (`zeMemAllocHost` pinned + per-device command queue и `zeCommandListAppendMemoryCopy`,
  host-sync через `zeCommandQueueSynchronize`). Междевайсный путь на этой машине
  только через host (~4–5 GB/s end-to-end на 16–128 MB; ~80 µs host-sync floor
  на малый вызов). Peer/IPC/кросс-shared USM — NA. S04 коллективы ещё **не
  реализует** (только k engines + compile + props); staging-транспорт — S08/S12.
  Детали — docs/notes/l0-spike.md §8.
- **Осторожно** (перенос из S02): counter-based `zexCounterBasedEvent*` —
  device-side примитив; `zeEventHostSignal` на нём даёт UNSUPPORTED_FEATURE, а
  host-query всегда SUCCESS. Classic `zeEventCreate` из ctypes **крашнул** драйвер
  (S02). S03 обошелся без событий вообще: `zeCommandQueueSynchronize` —
  рабочий host-sync. Для device-side швов — event pool, аккуратно (S08/S12).
- Находки S03 (не реализовывать до своих этапов; детали — docs/notes/l0-spike.md §8):
  1. Междевайсный путь только через host: staging end-to-end ~4–5 GB/s (16–128 MB);
     одна сторона d2h/h2d ~7–10 GB/s; на ≤1 MB числа кэш-шумят (не PCIe).
  2. `zeMemAllocShared` **не** кросс-девайсный на этой машине: копия между
     shared(ctx0/dev0) и dev1 → `OUT_OF_DEVICE_MEMORY`. Как транспорт не годится.
  3. `zeCommandListAppendMemoryCopy` на regular list + in-order queue +
     `zeCommandQueueSynchronize` — рабочий host-sync без событий; host-sync floor
     ~60–90 µs/вызов (аргумент за device-side синхронизацию швов, S08/S12).
  4. Host USM, аллоцированный в ctx0, доступен обеим картам (h2d host→dev1 из
     ctx0 работает) — база для единого staging-буфера ранга-агрегатора.
- Находки S02 (не реализовывать до своих этапов; детали — в docs/notes/l0-spike.md):
  1. `zeDeviceGetP2PProperties` в SDK 1.33.1 не имеет флага `ATOMICS_64BIT`
     (только ACCESS/ATOMICS) — учесть в дизайне allreduce.
  2. Драйвер рапортует `ZE_IPC_PROPERTY_FLAG_MEMORY|EVENT_POOL`, но фактически
     IPC не работает на Windows — не доверять `zeDriverGetIpcProperties`.
  3. `zeLoader` не экспортирует `zex*` напрямую — только через
     `zeDriverGetExtensionFunctionAddress` (важно для загрузки в плагине, S04/S12).
  4. `ZEX_mem_ipc_handles` документирован как Linux-only (official doc).
- Находки S01 (сохранены ниже, не реализовывать до своих этапов):
  1. `tools/continuous_batching` в ЭТОМ репо отсутствует — GenAI CB живёт в
     отдельном репо openvino.genai; в дереве openvino строка
     MODEL_DISTRIBUTION_POLICY только в properties.hpp (+pybind). Учесть при S13.
  2. `zeDeviceCanAccessPeer`/`zeDeviceGetP2PProperties`/`zeMemGetIpcHandle`/
     `zeMemOpenIpcHandle` НЕ объявлены в дереве intel_gpu — S02 брал системные
     L0 headers/loader. `zexCounterBasedEventGetIpcHandle/OpenIpcHandle` есть
     (zex_event.h:46-50, zex_common.h:167-204) и в плагине не используются.
  3. CPU-механика TP делает только N-split FC, K-split/allreduce отсутствует —
     allreduce(sum) для GPU проектируется с нуля (S08); CPU `split_parts` даёт
     неравные хвосты (len%n) — для int4 K-границ НЕ годится (K кратно 128).
  4. Counter-based events отключены на xe3/xe3p/unknown (ze_stream.cpp:246-250);
     на A770 (Xe-HPG) counter-based `Create2`/локальные события работают (S02),
     но IPC-экспорт их недоступен.
  5. Blob-cache GPU имеет compat-descriptor runtime requirements
     (plugin/compiled_model.cpp:97-120) — per-rank кэш различать по (rank, device)
     при S16.
