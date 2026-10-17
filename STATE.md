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
- [x] S07 Stage2.2 N-split + all-gather (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S08 Stage2.3 K-split + allreduce (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S09 Stage2.4 int4_ASYM (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
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

| 2026-10-17 | S07 Stage2.2 N-split + all-gather | **частично** (код+тест готовы) | **Rank в пасс — Вариант A**: новые internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks` (`TP_RANK`/`TP_NUM_RANKS`, `OV_CONFIG_RELEASE_INTERNAL_OPTION`, default -1/1) в `internal_properties.hpp`+`options.inl`; `TPCoordinator::compile_model_tensor_parallel` штампует их в per-rank `ExecutionConfig` через `set_property` (прямое значение переживает `clone()`+`finalize()` в `clone_and_transform_model`); `TransformationsPipeline::apply` строит `TPShardTransform(config.get_tp_rank(), config.get_tp_num_ranks())`. **Срез**: пасс теперь при `rank>=0 && num_ranks>1` для каждого `N_SPLIT`-узла с константными весами заменяет Constant на слайс по N-оси (для `MatMul` — по `transpose_b`; `[K,n_r]`/`[N,n_r]`), режет bias (операнд 2) тем же диапазоном, заполняет `TPSeam.split_axis=-1`/`shard_dim_size=N`, возвращает `true`. Слайс — на OV-уровне, до фьюжнов и до `propagate_constants` (upload), на устройство едет только `[K, N/ranks]`. Sub-byte (int4) явно отклоняется (`int4 = S09`). **Fallback**: `N % num_ranks != 0` ИЛИ потребитель слайса не element-wise/Result → узел понижается до `REPLICA` (полные веса) — корректность сохраняется, жёсткого throw нет (задокументировано). **All-gather-эквивалент (host-side)**: `TPShardTransform` дополнительно строит `std::vector<TPOutputSeam>` (rt_info `intel_gpu_tp_output_seams`) — индексы model-output, достижимые из срезанного узла только через element-wise (Add/Mul/Relu/Gelu/…); `TPPlan.output_seams` собирается на ранге-0. `TPCompiledModel::build_output_ports()` публикует для таких выходов СИНТЕТИЧЕСКИЙ порт полной формы `[.., N]` (Parameter+Result, хранятся в объекте) — контракт S05 «выходы только rank-0» сохранён. `TPInferRequest` теперь владеет выходными host-тензорами (`m_output_tensors`), после `infer()` собирает полный выход конкатенацией регионов `[r*N/k,(r+1)*N/k]` (`copy_last_axis_region`, без арифметики), реплики копирует с rank-0. Тесты S07: `NsplitMlpParityMatchesSingleGpu` (MatMul[8,16]→Relu→Result, TP2 vs 1-GPU, cos>0.999, full shape [1,16]), `NsplitNotDivisibleFallsBackToReplica` (N=15 → REPLICA, паритет). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. S04/«Блокеры») |

| 2026-10-17 | S08 Stage2.3 K-split + allreduce | **частично** (код+тест готовы) | **K-split весов**: `TPShardTransform` для `K_SPLIT` с константными весами (2D) режет Constant по K-оси (`find_weight_k_axis`: `transpose_b=false`→axis 0 `[K,N]`, `true`→axis 1 `[N,K]`), `offset=rank*K/ranks`, `K % ranks == 0`, элемент ≥8 бит (int4 → S09). **K-срез активации**: A-операнд (input 0) режется по последней оси через `ov::op::v8::Slice` (start/stop/step/axis=-1) — каждый ранг считает истинную частичную GEMM-сумму `A_r @ W_r` (не «A_full @ W_r», который размерно некорректен). **Численный инвариант**: выбирается вариант (б) из промпта (срез и весов, и активации); K-границы кратны `group_size` при int4 (S09), на S08 `group_size=1`. **Replicated additive terms**: `scan_k_reduction` проходит по потребителям K-узла через elementwise и собирает `Add`-слагаемые, которые реплицированы на каждом ранге: Constant (bias) и/или прямой вход-`Parameter` (residual). Истинный residual, произведённый внутри графа (не Constant/не Parameter), или не-elementwise потребитель → узел НЕ режется (корректность требует device-side allreduce ДО Add — S09+); классификация (`seam.kind=K_SPLIT`) при этом сохраняется (стабильность `tp_seams`). **Allreduce(sum)**: `TPInferRequest::assemble_reduce_sum` суммирует per-rank выходы в fp32 (PLAN §7.4), затем вычитает `(num_ranks-1)*term` для каждого replicated-слагаемого (bias/residual добавляются ровно один раз), затем кастует к выходному типу. `TPOutputSeam.collective` + `replicated_adds`; новый RO-property `ov::intel_gpu::tp_output_seams`. `build_output_ports` решейпит только `ALL_GATHER`. Тесты: `KsplitResidualParityMatchesSingleGpu` (residual-Parameter + bias-Constant, TP2 vs 1-GPU cos>0.999, проверка `tp_output_seams==ALL_REDUCE_SUM`), `KsplitNotDivisibleFallsBackToReplica` (K=7, без среза, `tp_output_seams=={}`, паритет). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры») |

| 2026-10-17 | S09 Stage2.4 int4_ASYM | **частично** (код+тест готовы) | **Распознавание dequant-подграфа**: на момент `TPShardTransform` (до `MarkDequantization`/конвертации FC) int4-проекции — это `MatMul` с dequant-цепочкой на операнде 1: `weight(u4 [..,N,G,gs]) -> Convert(f16) -> Subtract(Convert(zp)) -> Multiply(scale) -> Reshape([..,N,K]) -> Convert(f32)`, поэтому `has_constant_weights` был false и int4-узлы были REPLICA (S06–S08). `match_quantized_weight` распознаёт эту форму (weight/scale/zp/reshape + group_axis/group_size/num_groups). **Срез**: N-split режет weight+scale+zp по N-оси целыми oc-чанками и обновляет target `Reshape` (`n_out→count`), плюс bias-Add-константы; K-split режет weight+scale+zp по group-оси целиком группами (`G % num_ranks == 0`, т.е. K кратно `group_size*num_ranks`) и обновляет `Reshape` (`K→count_g*gs`); активация режется как в S08 (`v8::Slice` по последней оси). **Sub-byte выравнивание**: `slice_is_byte_aligned` (row/off/slab_bits % 8) — правило именно `row_bits`, а не `inner_bits`; невыровненный/неделимый случай → REPLICA (без throw), классификация K_SPLIT сохраняется. zp per-tensor (один элемент) НЕ режется. Реальный Qwen остаётся в основном REPLICA (элементные цепочки достигают следующего GEMM → нужен inter-node all-gather, S10+; o_proj/down_proj residual внутренний → K-split небезопасен, S08-ограничение). Тесты: `Int4NsplitParityMatchesSingleGpu`, `Int4KsplitParityMatchesSingleGpu` (cos>0.999 + проверка `tp_output_seams`), `Int4KsplitIndivisibleFallsBackToReplica` (G=9, без all-reduce). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры») |

## Блокеры / открытые вопросы
- **Сборка OpenVINO невозможна в этой среде (2026-10-17, S04).** Тулчейн на месте:
  MSVC `cl` 14.44 (VS 2022 Community), `cmake` 4.0.3, `ninja` — но все git-сабмодули
  не инициализированы (пустые каталоги). `cmake -G Ninja -S . -B build_tp ...` падает на
  `thirdparty/ittapi` (`ittapi::ittnotify does not represent a cmake target`), а также
  отсутствуют `gtest`, `json/nlohmann_json`, `ocl/{cl_headers,clhpp_headers,icd_loader}`,
  `flatbuffers`, `intel_gpu/thirdparty/onednn_gpu` и др. `git submodule update --init`
  = сетевой fetch (запрещён remote-политикой) и многогигабайтная закачка. Итог: S04 GATE
  подтверждён только статически (ревью кода/сигнатур); сборка/запуск теста — вне сессии.
  Актуально для S05, S06, S07, S08 и S09: GATE подтверждён по коду/плану (ревью), прогон `tensor_parallel`
  тестов — вне сессии.
- **Решено (2026-10-17): remote не используем.** Сессия работает исключительно в
  локальном репозитории `C:\openvino-tp`; на `origin`/`upstream` ничего не пушим и
  не изменяем. Формулировка «push» в старых записях/промптах считается снятой.
  Локальные коммиты S01/S02: 7bd6c6d4, d0d2a20c, b97946ad, e6cbc7d5 (и далее).
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- **S09 done (частично, 2026-10-17)** — Stage 2.4 int4_ASYM. Артефакты/решения:
  - **Что видит пасс**: `TPShardTransform` вставлен на `transformations_pipeline.cpp:660` — ДО
    `MarkDequantization` (`:701`) и до конвертации `MatMul`→`FullyConnectedCompressed` (`:1758/:1763`).
    Значит, int4-проекции на входе пасса — это `MatMul`, у которого операнд 1 = dequant-подграф
    (НЕ `FullyConnectedCompressed`). Проверено по `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml`.
  - **Форма dequant-подграфа** (эталонно задокументирована в `docs/notes/int4-asym-weights.md`):
    `weight(u4 [..,N,G,gs]) -> Convert(f16) -> Subtract(Convert(zp u4 [..,N,G,1])) -> Multiply(scale f16
    [..,N,G,1]) -> Reshape([..,N,K]) -> Convert(f32) -> MatMul(transpose_b=true)`. Пример Qwen `in_proj_qkv`:
    weight u4 `[8192,32,128]` (N=8192,K=4096,G=32,gs=128), zp u4 `[8192,32,1]`, scale f16 `[8192,32,1]`.
  - **`match_quantized_weight`** (в `tp_shard_transform.cpp`) идёт от операнда 1 вглубь до Constant(u4),
    собирая `weight/scale/zero_point/reshape`; `group_axis = rank-2`, `group_size = last dim`,
    `num_groups = weight.shape[group_axis]`. `has_constant_weights` теперь принимает и bare-Constant,
    и dequant-подграф → int4-узлы корректно классифицируются (N_SPLIT/K_SPLIT вместо REPLICA).
  - **Срез N-split int4** (`apply_nsplit_quantized`): режет weight по N-оси (== 0 для `[N,G,gs]`) целыми
    oc-чанками; scale/zp — по их N-оси; пересобирает упомянутую в `Reshape` N (`n_out→n_out/ranks`) новым
    i64-Constant; bias-Add-константы режутся по последней оси. `n_axis != group_axis` и `!= last`.
  - **Срез K-split int4** (`apply_ksplit_quantized`): режет weight/scale/zp по group-оси ЦЕЛЫМИ группами,
    `G % num_ranks == 0` (⇔ K кратно `group_size*num_ranks`); `Reshape` target `K→count_g*gs`; активация —
    как в S08 (`slice_last_axis` по последней оси, offset=`rank*count_g*gs`). zp per-tensor (1 элемент)
    НЕ режется (`shape_size==1` → keep as-is).
  - **Правило выравнивания sub-byte** (важно!): байтовое выравнивание проверяется как
    `row_bits % 8 && off_bits % 8 && slab_bits % 8`, где `row_bits = shape[axis]*inner*bitwidth`
    (`inner` = произведение осей ПОСЛЕ axis). НЕ `inner_bits%8` (это была бы ошибка). Невыровненный срез →
    fallback REPLICA (без throw); `slice_constant` бросает `OPENVINO_ASSERT` только при прямом неверном вызове.
  - **`is_sliceable_type`** теперь допускает любую статическую ширину >0 (было ≥8 бит) — выравнивание
    энфорсит `slice_is_byte_aligned`. Это безопасно для bare-весов ≥8 бит.
  - **Безопасность потребителей (обновлено)**: `scan_nsplit_consumers` (заменил `consumers_are_slice_safe`) —
    BFS по elementwise: не-elementwise потребитель (следующий GEMM) → unsafe; полновесный непостоянный Add
    (residual) → unsafe; Add-Constant шириной N → собирается в `side_constants` и режется. `big` Qwen
    qkv/gate/up достигают другого GEMM → REPLICA (нужен inter-node all-gather, S10+). o_proj/down_proj
    residual внутренний → K-split unsafe (S08). Т.е. реальный Qwen GATE пока недостижим → GATE частичный.
  - **`is_sliceable_type`/bare-путь S07/S08 не сломан**: `apply_nsplit_bare`/`apply_ksplit_bare` теперь
    также проверяют `slice_is_byte_aligned` (для ≥8 бит всегда true).
  - **Артефакты**: `src/plugin/transformations/tp_shard_transform.cpp` (главные правки),
    `docs/notes/int4-asym-weights.md` (новый конспект), тесты в `tensor_parallel.cpp`
    (`Int4NsplitParityMatchesSingleGpu`, `Int4KsplitParityMatchesSingleGpu`,
    `Int4KsplitIndivisibleFallsBackToReplica` + билдеры `int4_asym_dequant_weights`/`make_int4_*`).
    CMake не правился: `src/plugin/**/*.cpp` и тесты подхватываются GLOB.
  - **ОГРАНИЧЕНИЯ S09 (важно для S10+):**
    1. Inter-node all-gather между двумя GEMM-ами (N-split → следующий GEMM) НЕ реализован → большинство
       реальных Qwen-проекций остаются REPLICA. Это S10+ (или требует device-side коллективов).
    2. Внутренний residual (`x + o_proj(attn)`) по-прежнему не K-сплитится (нужен device-side allreduce
       ДО Add) — как S08.
    3. `match_quantized_weight` не разбирает `Transpose` в цепочке (нестандартный layout → fallback REPLICA).
    4. Групповые границы предполагают континуальный K-индекс (gs-блоки идут подряд по K) — стандартный
       int4_ASYM OV-layout; иные упаковки не поддержаны.
  - **Заметка S09→S10**: S10 = per-rank KV + GQA heads + агрегация state. Точки: `TPInferRequest::query_state`
    сейчас только rank-0; per-rank `Stateful`/KV нужно собирать/агрегировать. GATE — greedy 32-token
    паритет против 1-GPU на Qwen3.5-9B-int4. После S10 реальные attention-проекции могут начать резаться
    (нужен all-gather), что сделает полный Qwen GATE возможным.
- **S08 done (частично, 2026-10-17)** — Stage 2.3 K-split + all-reduce(sum). Артефакты/решения:
  - **Выбор по срезу активации — вариант (б)**: и веса, и активация режутся по K. Прямой вариант (а)
    («A_full @ W[:,k_r]») размерно некорректен (несовпадение K), поэтому отвергнут. Веса — `slice_constant`
    по K-оси; активация — `ov::op::v8::Slice` по последней оси (`axis=-1`, `start=offset`, `stop=offset+count`),
    вставляется в граф (работает и для `Parameter`, и для промежуточных тензоров). x-срез не меняет контракт
    входов (пользователь даёт полный тензор, режется в графе per-rank).
  - **Геометрия K**: `find_weight_k_axis` (2D-веса обязательны) — `transpose_b=false` → ось 0 (`[K,N]`),
    `transpose_b=true` → ось 1 (`[N,K]`); `K = weights.shape[k_axis]`; срез при `K % num_ranks == 0` и
    `act_last_dim == K`; элемент ≥ 8 бит. `group_size` учитывается только в S09.
  - **Механизм allreduce — host-side, subtractive** (S08-компромисс, device-side — S12): каждый ранг
    считает `partial_r + term` (term = replicated bias/residual), reducer делает
    `full = sum_r(partial_r + term) - (num_ranks-1)*term = sum_r partial_r + term`. Это даёт «term один раз»
    без удаления Add/Constant из графа. fp32-аккумуляция обязательна (PLAN §7.4).
  - **Replicated terms**: `scan_k_reduction` собирает из elementwise-цепочки после K-узла: `Add(K, Constant)`
    → CONSTANT (broadcast-expanded к форме выхода на compile-time); `Add(K, Parameter)` → INPUT (индекс
    параметра; broadcast по хвостовым осям в runtime). Формула `TPReplicatedAdd`. Любой другой Add-операнд
    (истинный residual, `Add(mm,mm)`) или не-elementwise потребитель → узел НЕ режется (safe=false).
  - **Стабильность `tp_seams`**: при отказе от среза `seam.kind`/`collective` СОХРАНЯЮТСЯ (не деградируют
    в REPLICA) — иначе сломается S06-тест `ShardTransformClassifiesResidualProjectionAsKSplit`
    (`Add(matmul,matmul)` → нережущийся K_SPLIT). Не-срезанный узел = реплика (rank0 считает полный GEMM),
    `tp_output_seams` пуст → сборка «replicated copy» → корректно.
  - **Новый property** `ov::intel_gpu::tp_output_seams` (RO, `vector<string>`, формат
    `COLLECTIVE:producer:split_axis:shard_dim_size:repl_adds`); `TPCompiledModel`/обычный/плагин отдают
    согласованно (пусто вне TP). `tp_seams` формат НЕ менялся.
  - **Артефакты**: `tp_plan.{hpp,cpp}` (`TPReplicatedAdd`/`TPReplicatedAddKind`, `TPOutputSeam.collective`+
    `replicated_adds`, encoder), `tp_shard_transform.cpp` (K-ветка + `find_weight_k_axis`/`slice_last_axis`/
    `constant_to_floats`/`broadcast_expand`/`scan_k_reduction` + `is_sliceable_type` guard и на N-split),
    `tp_compiled_model.cpp` (build_output_ports только ALL_GATHER + tp_output_seams), `tp_infer_request.{hpp,cpp}`
    (`assemble_reduce_sum`, `tensor_to_floats`/`floats_to_tensor`/`accumulate_scaled_broadcast`),
    `internal_properties.hpp`, `compiled_model.cpp`/`plugin.cpp` (property), тесты в `tensor_parallel.cpp`.
  - **ОГРАНИЧЕНИЯ S08 (важно для S09+):**
    1. Истинный residual (не Constant/Parameter, а результат внутри графа) НЕ поддержан → узел реплицируется.
       Реальные трансформеры (`x = x + o_proj(attn)`) именно такие → на S08 их o_proj/down_proj не K-сплитятся,
       пока device-side allreduce не будет вставлен ДО Add (S09+ или S12). GATE-модель использует residual-
       `Parameter`/bias-`Constant` — это честно ограниченный, но корректный сценарий.
    2. Срез активации по K конфликтует с N-split, если A — выход N-split узла (цепочка N→K). Комбинированный
       сценарий не реализован (S08 не строит реальную all-gather между узлами) — задокументировано.
    3. `slice_last_axis` требует, чтобы вход-активация имела статический ранг и статическую K-ось; иначе узел
       реплицируется.
  - **Заметка S08→S09 (int4_ASYM)**: K-границы должны быть кратны `group_size=128`; `slice_constant` сейчас
    явно отклоняет sub-byte (`is_sliceable_type`), поэтому int4 K_SPLIT-узлы на S08 просто реплицируются
    (не падают). S09: резать веса+scales+zp по group-границам, K кратно 128, N целыми oc-группами. GATE —
    Qwen3.5-9B-int4 TP2 prefill паритет.
- **S07 done (частично, 2026-10-17)** — Stage 2.2 N-split + all-gather-эквивалент. Артефакты/решения:
  - **Выбор механизма rank = Вариант A** (per-rank config): `ov::intel_gpu::tp_rank`/`tp_num_ranks`
    (RELEASE_INTERNAL) в `include/intel_gpu/runtime/internal_properties.hpp` + `options.inl`;
    `TPCoordinator` штампует `set_property(...)` в `RankContext::config` до `clone_and_transform_model`;
    `TransformationsPipeline::apply` → `register_pass<TPShardTransform>(config.get_tp_rank(),
    config.get_tp_num_ranks())`. Мотив: сохраняет позицию «до фьюжнов»; TP-off не затронут
    (`tp_rank=-1`/`tp_num_ranks=1` → пасс не срезает). Прямая опция (не user-property) переживает
    `config.clone()`+`finalize()` (finalize чистит только user-properties; значение опции копируется
    конструктором копирования `ExecutionConfig`).
  - **Срез**: `TPShardTransform::run_on_model` теперь меняет граф и возвращает `true`. Для `N_SPLIT`
    с константными весами: `slice_constant` по N-оси (ось определяется `transpose_b` для
    MatMul/FC, fallback — совпадение с последней dim выхода), `offset = rank*N/ranks`,
    bias (input 2) — тем же диапазоном; `TPSeam.split_axis=-1`/`shard_dim_size=N`. НЕ режет K-ось,
    не трогает `K_SPLIT`/`REPLICA`, не режет sub-byte (int4 → S09, явный throw в хелпере).
  - **Деградация вместо throw** (задокументированный выбор негатива): невыполненные условия
    (N % ranks != 0, динамический N/веса, небезопасный потребитель) → `seam.kind=REPLICA`, полные веса,
    без сборки. Хелпер `slice_constant` при прямом неверном вызове бросает `OPENVINO_ASSERT`.
  - **All-gather = host-конкатенация** (S07-компромисс): `TPOutputSeam{output_index,split_axis=-1,
    shard_dim_size,producer}` (rt_info `intel_gpu_tp_output_seams` → `TPPlan.output_seams`);
    `TPCompiledModel` публикует синтетический полный порт только для этих выходов;
    `TPInferRequest::assemble_outputs()` конкатенирует регионы из per-rank выходов (без суммирования).
    Device-side/коллективный транспорт — S08/S12.
  - **ОГРАНИЧЕНИЕ S07 (важно для S08):** полная сборка корректна ТОЛЬКО когда срез-узел достигает
    model-output через purely element-wise операции (или напрямую). Цепочка `N-split → (elementwise) →
    другой GEMM` требует РЕАЛЬНОЙ all-gather между узлами (полный вход K для второго GEMM) — это вне
    S07 (пасс понижает узел до REPLICA, если у него не-elementwise потребитель; при elementwise-цепочке
    в следующий GEMM сработает shape-inference ошибка). Реальные Qwen qkv (reshape/rope/sdpa — не
    elementwise) → пока компилируются как REPLICA.
  - **Артефакты**: правки в `tp_shard_transform.{hpp,cpp}`, `tp_plan.{hpp,cpp}` (TPOutputSeam,
    `find_output_seam`, rt_info-хелперы), `tp_coordinator.cpp`, `tp_compiled_model.{hpp,cpp}`,
    `tp_infer_request.{hpp,cpp}`, `transformations_pipeline.cpp`, `internal_properties.hpp`, `options.inl`;
    тесты в `tensor_parallel.cpp`.
  - **Заметка S07→S08:** S08 = K-split + allreduce(summ, fp32-acc) + швы-события. Точки:
    `TPSeam.kind==K_SPLIT` (уже классифицируется как residual-проекция) — веса резать по K-оси (для
    `MatMul` `transpose_b=false` веса `[K,N]` → срез по axis 0; для int4 K кратно group_size=128),
    bias НЕ резать (добавляется один раз в reducer), `TPCollectiveKind::ALL_REDUCE_SUM`. Сейчас K_SPLIT
    узлы копируются целиком на каждом ранге (REPLICA-эквивалент) — S08 впервые срежет их и введёт
    суммирование частичных результатов (host-side на S08 достаточно, device-side — S12). GATE — паритет
    на тесте с residual-проекцией.
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
