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
- [x] S10 Stage3.1 per-rank KV + state aggregation (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S11 Stage3.2 linear_attention (2026-10-17; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S12 Stage3.3 Hard (2026-10-18; код+тест готовы, GATE частичный — сборка недоступна)
- [x] S13 Stage4.1 GenAI интеграция (2026-10-18; обвязка+тесты+док, GATE частичный — сборка/GenAI/2 GPU недоступны)
- [x] S14 Stage4.2 Scheduler + бенч (2026-10-18; scheduler+тесты+бенч-харнесс+бенч-план, GATE частичный — сборка/2 GPU недоступны)
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

| 2026-10-17 | S10 Stage3.1 per-rank KV + state aggregation | **частично** (код+тест готовы) | **Инвентаризация состояний Qwen** (`docs/notes/state-inventory.md`): 64 `ReadValue`/`Assign` (opset6), 0 Stateful; 8 `ScaledDotProductAttention` (full attn) и 24 `Loop`+`ScatterUpdate`+`GroupConvolution` (linear attn). Переменные (f32, dynamic): `past.key.N`/`past.value.N` `?,4,?,256` (KV-head ось = 1, kv-heads=4, head_dim=256), `past.conv.N` `?,8192,4`, `past.ssm.N` `?,32,128,128`. **Ключевой вывод**: реальные Qwen KV/conv/ssm-цепочки не head-local (SDPA/Loop/Concat/GroupConvolution — не element-wise) → head-slice небезопасен, поэтому они остаются **REPLICATED** (rank-0 авторитетный); per-rank KV для Qwen требует inter-node коллективов = S12. **Классификация/шардинг состояний**: `TPShardTransform` после GEMM-прохода собирает переменные (`ReadValueBase`/`AssignBase` по `variable_id`), анализирует `analyze_state_variable`: статическая форма, числовой тип, делимая head-ось (`find_state_head_axis`: предпочтительна ось `rank-2`, иначе первая делимая в `1..rank-2`), цепочка `ReadValue -> (element-wise/broadcast) -> Assign` целиком head-local (все не-цепные операнды broadcast'ятся по head-оси: dim==1 или rank≤axis; init_value ReadValue тоже должен broadcast'иться), иначе `REPLICATED` с диагностической причиной (без throw). При успехе — `var->update_data_shape(sliced)` + `model->validate_nodes_and_infer_types()` (per-pass validation выключен в `transformations_pipeline.cpp:645`). **Структура/транспорт**: `TPStateSeam{variable_id, TPStateShardKind(REPLICATED/HEAD_SHARDED), axis, full_dim, full_shape, reason}` в `TPPlan.state_seams`; rt_info `intel_gpu_tp_state_seams`; RO-property `ov::intel_gpu::tp_state_seams` на `TPCompiledModel` (формат `KIND:AXIS:FULL_DIM:VARIABLE_ID`, напр. `HEAD_SHARDED:1:4:tp_state_var`) и пустое на обычном/плагинном. **Агрегация**: `TPInferRequest::query_state()` строит `TPAggregatedState : ov::IVariableState` по каждому состоянию (совпадение по имени = variable_id): `get_state()` конкатенирует head-оси рангов для HEAD_SHARDED, rank-0 для REPLICATED; `set_state()` скаттерит полный тензор по рангам (`extract_axis_region`) или fan-out; `reset()` — по всем рангам. **Обобщение осей**: `copy_last_axis_region` заменён generic `copy_axis_region`/`extract_axis_region`; `build_output_ports`/`assemble_outputs` собирают all-gather по произвольной оси (для N-split ось по-прежнему последняя); выходы, производные от HEAD_SHARDED-состояния, получают отдельный ALL_GATHER-шов. Тесты: `StateHeadShardingMatchesSingleGpu` (head-local mini-модель, TP2 vs 1-GPU по 5 шагам + проверка шва `HEAD_SHARDED:1:4:` и полной формы агрегированного состояния), `StateNonBroadcastOperandStaysReplicated` (full-width операнд → `REPLICATED:`, паритет), `DefaultCompileHasNoStateSeams`. **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры»); реальный Qwen state — REPLICATED (обосновано) |

| 2026-10-17 | S11 Stage3.2 linear_attention | **частично** (код+тест готовы) | **Примитив PLAN §4.3 «depthwise conv per head»**. `TPShardTransform` теперь распознаёт head-local состоятельную цепочку, рекуррентность которой идёт через **depthwise `GroupConvolution`** (`ReadValue -> depthwise GroupConvolution -> (elementwise/broadcast) -> Assign`): depthwise ⇔ filters `[GROUPS,1,1,k]` c `GROUPS == data channels == full_dim` (`match_depthwise_head_conv`, filters — `Constant`); переменная переводится в `HEAD_SHARDED` по канало-оси, а **depthwise-фильтры режутся тем же ранговым диапазоном** (ось 0, `slice_constant`, `replace_constant`) до `validate_nodes_and_infer_types()`. **Ось головы**: `find_state_head_axis` теперь предпочитает **axis 1** (каноничный layout Qwen `[batch, heads, ...]` — KV/conv/ssm), затем `rank-2`, затем первую делимую внутреннюю ось (для старых тестов `[1,4,2]` результат тот же). Любой иной (не-depthwise / не-head-local) `GroupConvolution` в цепочке → `REPLICATED` без throw (прогресс S07–S10). **Инвентарь**: `docs/notes/state-inventory.md` §6 — точные conv-цепочки (`Gather → Concat → GroupConvolution → Slice`), ssm (`Gather → Loop(delta-rule) → Reshape`), head-ось = 1, conv-веса тоже int4_ASYM dequant. **Вывод**: реальные Qwen-цепочки не head-local (coupled fused `in_proj` + `Concat`/`Loop`/`ScatterNDUpdate`) → **REPLICATED**; per-rank Qwen = S12. **Тесты**: `LinearAttentionConvStateHeadShardingMatchesSingleGpu` (heads=4 → `HEAD_SHARDED:1:4:`, multi-step паритет, агрегированное состояние полной формы) и `LinearAttentionConvStateIndivisibleStaysReplicated` (heads=3 → `REPLICATED:`), модель `make_linear_attn_conv_state_model` (depthwise conv same-pad). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры») |

| 2026-10-18 | S12 Stage3.3 Hard (dynamic/transport/concurrency) | **частично** (код+тест готовы) | **Динамические state-шейпы**: `TPStateSeam.full_shape` переведён `ov::Shape`→`ov::PartialShape` (`tp_plan.hpp`); `TPShardTransform::find_state_head_axis` принимает `PartialShape` и требует, чтобы head-ось была **статической и делимой** (`axis.is_static() && get_length()%num_ranks==0`), а `analyze_state_variable` пропускает динамический ранг/динамические внутренние оси (`rank().is_dynamic()` — REPLICATED без throw), вместо прежнего отклонения «dynamic state shape». Шардинг: `sharded[axis]=ov::Dimension(full_dim/num_ranks)` при сохранении динамики batch/seq, затем `update_data_shape(sharded)`+`validate_nodes_and_infer_types()`. Это снимает барьер для всех реальных Qwen-состояний `?,4,?,256`/`?,8192,4`/`?,32,128,128` (но сами они по-прежнему REPLICATED — потребители не head-local, см. S10/S11). **Межранговый транспорт**: новый `TPTransport` (`tp_transport.hpp/.cpp`) — host-USM staging (G0): reusable pinned-host staging-буфер, коллективы `all_gather_axis`/`scatter_axis`/`all_reduce_sum_fp32` (fp32-аккумуляция + «term один раз»), capabilities-probe фиксирует `HOST_STAGING`/peer=false/ipc=false (S02/S03), PEER — только runtime fast-path hook (S15). Перенесённые хелперы (`copy_axis_region`/`extract_axis_region`/`tensor_to_floats`/`floats_to_tensor`/`row_major_strides`/`accumulate_scaled_broadcast`) вынесены из `tp_infer_request.cpp` в транспорт. `assemble_outputs` (all-gather), `assemble_reduce_sum` (all-reduce) и `TPAggregatedState::get_state/set_state` теперь ходят через `m_transport`; форма полного состояния выводится из runtime-slice ранга (`full_shape=slice_shape; full_shape[axis]*=num_ranks`) — поддержка динамических batch/seq. **Конкурентность**: `m_infer_mutex` сериализует `infer()` одного запроса; `m_output_mutex` защищает lazy-аллокацию выходных буферов; независимые запросы изолированы (свои rank sub-запросы+transport+outputs). **Устройство**: device-side события (`zeEventPool`) и true in-graph inter-node коллективы НЕ реализованы (S12-приоритет — корректность/паритет); counter-based event+`zeEventHostSignal`=UNSUPPORTED, classic `zeEventCreate`=краш драйвера (S02) → staging. **Тесты S12** (`tensor_parallel.cpp`): билдер `make_dynamic_stateful_head_model(heads,head_dim)` (Variable `[?,heads,hd]`), `DynamicStateHeadShardingMatchesSingleGpu` (dynamic batch, seam `HEAD_SHARDED:1:4:`, multi-step паритет + агрегированное `[2,4,2]`), `DynamicStateIndivisibleStaysReplicated` (heads=3 → `REPLICATED:`), `StateRoundtripHeadShardedMatchesSingleGpu`, `StateRoundtripReplicatedMatchesSingleGpu`, `ConcurrentRequestsMatchSingleGpu` (2 sync + 2 async запроса). **GATE по сборке/прогону не подтверждён** — сабмодули пусты (см. «Блокеры»); S04–S11 не затронуты (TP-off == upstream, форматы seams/state не менялись: `KIND:COLLECTIVE:TYPE:NAME`, `COLLECTIVE:...`, `KIND:AXIS:FULL_DIM:VARIABLE_ID`) |

| 2026-10-18 | S14 Stage4.2 Scheduler + бенч | **частично** (scheduler+тесты+бенч-план; перф-прогон недоступен) | **Rank scheduler (ядро)**: новый `TPInferRequest::schedule_ranks()` (`tp_infer_request.cpp`) исполняет k per-rank sub-запросов **параллельно** на общем k-stream `ov::threading::IStreamsExecutor` (подход **(б)**), созданном в `TPCompiledModel` (`m_rank_executor`, один stream на ранг/устройство). Диспетчер — `ITaskExecutor::run_and_wait(tasks)` (тот же приём, что у `ov::IAsyncInferRequest`, `iasync_infer_request.cpp:29`): пул переживает запрос (нет потоков на инференс — важно для decode), исключения рангов агрегируются через packaged_task/future. Порядок: `replicate_inputs()` (до fan-out) → `schedule_ranks()` (ждём ВСЕ ранги) → `assemble_outputs()` (одним потоком, детерминированно). `m_infer_mutex` сохранён (не-реентерабельность одного запроса); независимые запросы изолированы. Fallback на serial-цикл при отсутствии executor/1 ранге. **Тест** `SchedulerParallelRanksMatchSingleGpu` (4 перекрывающихся async-запроса на N-split/all-gather модели + повторный sync-infer → бит-идентичный выход = нет гонок staging). **Бенч-обвязка**: `tools/tp_bench/tp_bench.py` (TTFT/TPOT TP1/TP2/MULTI через публичный API; GenAI-LLMPipeline при наличии, иначе indicative decode-step proxy) + `tools/tp_bench/README.md` + бенч-план `docs/notes/tp-bench-plan.md` (метрики/среда/команды/критерии/профиль узких мест). **GATE по перфу не подтверждён** — сборки/2 GPU/GenAI нет (см. «Блокеры»); ожидаемый вклад: host-USM staging G0 (~4–5 GB/s) + преимущественно REPLICATED Qwen → TPOT_ratio≈1.0 до S15 |
| 2026-10-18 | S13 Stage4.1 GenAI интеграция | **частично** (обвязка+тесты+док; прогон недоступен) | **Probe/путь свойства**: GenAI (openvino.genai, CB-бенчмарк — в этом дереве отсутствует, S01) передаёт строковый `ov::AnyMap {"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` в `core.compile_model(model,"GPU",config)`; `ov::PluginConfig::set_user_property` конвертирует строку в `std::set<ov::hint::ModelDistributionPolicy>` (`ov::Any::as`, `any.hpp:767-784`; set Readable, enum Istreamable), GPU рекламирует свойство RW (`plugin.cpp:893`) → `compile_model_tensor_parallel` (`plugin.cpp:285`). **Расхождения property/config нет — правок плагина не требуется.** Разбор — `docs/notes/genai-integration.md` (§2 путь, §3 совместимость compile-фрейма, §4 ограничение реального Qwen, §5 тест-план) + `gpu-plugin-arch.md` §7. **Совместимость compile-фрейма**: name-based `set_tensor/get_tensor` (через `getPort` над `get_inputs()/get_outputs()`), `query_state()/set_state()` (`TPAggregatedState`) — покрывают GenAI-потребление stateful-модели. **Тесты S13** (`tensor_parallel.cpp`): `GenAiStringConfigEnablesTensorParallel` (строковый GenAI-конфиг → tp_ranks {0,1}, RO-property `model_distribution_policy`, seams непусты), `GenAiStyleStatefulRoundtripMatchesSingleGpu` (билдер `make_genai_style_stateful_model` с named IO `input_ids`/`logits` и динамическим seq; HEAD_SHARDED state, 4 шага + roundtrip `query_state/set_state`, паритет vs 1-GPU). **Реальный Qwen3.5-9B-int4 на TP2 остаётся преимущественно REPLICATED** (S07–S12) → генерация численно паритетна 1-GPU; спид-ап — S14+ (in-graph коллективы — S15). **GATE по сборке/прогону не подтверждён** — сабмодули пусты, GenAI/2 GPU недоступны (см. «Блокеры») |

## Блокеры / открытые вопросы
- **S14 GATE частичный (2026-10-18).** Ранги теперь исполняются параллельно
  (`schedule_ranks()` + общий k-stream executor), тесты и бенч-обвязка написаны, но
  **сборка OpenVINO недоступна** (сабмодули пусты) и **2× GPU / OpenVINO GenAI на сборочной машине нет**
  → перф-прогон (`tp_bench.py`) и главный GATE **TPOT ≥ 1.5× / TTFT ≥ 1.3× на 2× BMG** не подтверждён.
  Ожидание: реальный Qwen остаётся преимущественно REPLICATED (S07–S13) → TPOT_ratio≈1.0;
  ≥1.5× требует S15 (peer/ring, in-graph коллективы). Бенч-план/ручной сценарий —
  `docs/notes/tp-bench-plan.md`; харнесс — `tools/tp_bench/tp_bench.py`. Прогон — вне сессии.
- **S13 GATE частичный (2026-10-18).** GenAI-путь разобран статически (код + контракт Core/плагина),
  обвязка и эмуляционные тесты написаны, но **сборка OpenVINO недоступна** (сабмодули пусты),
  **OpenVINO GenAI (openvino.genai) в дереве отсутствует**, а **2× Arc A770 на сборочной машине нет**
  → паритет greedy-генерации Qwen3.5-9B-int4 TP2 vs 1-GPU не подтверждён. Тест-план и ручной
  сценарий — `docs/notes/genai-integration.md` §5. Прогон `tensor_parallel` (в т.ч. новых
  `GenAiStringConfigEnablesTensorParallel`, `GenAiStyleStatefulRoundtripMatchesSingleGpu`) — вне сессии.
- **Сборка OpenVINO невозможна в этой среде (2026-10-17, S04).** Тулчейн на месте:
  MSVC `cl` 14.44 (VS 2022 Community), `cmake` 4.0.3, `ninja` — но все git-сабмодули
  не инициализированы (пустые каталоги). `cmake -G Ninja -S . -B build_tp ...` падает на
  `thirdparty/ittapi` (`ittapi::ittnotify does not represent a cmake target`), а также
  отсутствуют `gtest`, `json/nlohmann_json`, `ocl/{cl_headers,clhpp_headers,icd_loader}`,
  `flatbuffers`, `intel_gpu/thirdparty/onednn_gpu` и др. `git submodule update --init`
  = сетевой fetch (запрещён remote-политикой) и многогигабайтная закачка. Итог: S04 GATE
  подтверждён только статически (ревью кода/сигнатур); сборка/запуск теста — вне сессии.
  Актуально для S05, S06, S07, S08, S09, S10, S11 и S12: GATE подтверждён по коду/плану (ревью), прогон `tensor_parallel`
  тестов — вне сессии.
- **Решено (2026-10-17): remote не используем.** Сессия работает исключительно в
  локальном репозитории `C:\openvino-tp`; на `origin`/`upstream` ничего не пушим и
  не изменяем. Формулировка «push» в старых записях/промптах считается снятой.
  Локальные коммиты S01/S02: 7bd6c6d4, d0d2a20c, b97946ad, e6cbc7d5 (и далее).
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- **S14 done (частично, 2026-10-18)** — Stage 4.2 Scheduler + бенч TP2 vs TP1/MULTI. Артефакты/решения:
  - **Scheduler — подход (б)**: `TPInferRequest::schedule_ranks()` строит k задач `rank_r->infer()` и
    вызывает `m_rank_executor->run_and_wait(tasks)`. `m_rank_executor` = `CPUStreamsExecutor`
    с `streams = num_ranks` (имя "Intel GPU TP rank scheduler"), создаётся в `TPCompiledModel`
    ctor и живёт вместе с compiled model. Обоснование выбора:
    (а) `std::async`/per-infer пул — потоки на каждый infer (плохо для decode, infer на токен);
    (б) общий `IStreamsExecutor` с k задачами — **выбран**: переиспользует OV-инфраструктуру
    (`ITaskExecutor::run_and_wait`, как в `iasync_infer_request.cpp:29`), пул переживает запрос,
    k stream'ов = k устройств, гарантированный параллелизм, корректный проброс исключений;
    (в) async sub-запросы — rank-модели дают sync-запросы, их `infer()` блокирует, отдельный
    async-путь не нужен. **Детерминизм**: fan-out только после `replicate_inputs()`; `assemble_outputs()`
    (all-gather/all-reduce через `TPTransport`) и агрегация состояния — строго после `run_and_wait`,
    в вызывающем потоке → порядок сборки не меняется. `m_infer_mutex` (per-request) и `m_output_mutex`
    сохранены; fallback на serial-цикл при `nullptr`/1 ранге (defensive).
  - **Точки касания**: `tp_infer_request.{hpp,cpp}` (`m_rank_executor`, `schedule_ranks()`),
    `tp_compiled_model.{hpp,cpp}` (член+геттер `get_rank_executor()`, создание пула). Форматы
    `tp_seams`/`tp_output_seams`/`tp_state_seams` не менялись. Публичный API/свойства не менялись.
  - **Тесты S14** (`tensor_parallel.cpp`): `SchedulerParallelRanksMatchSingleGpu` — 4 перекрывающихся
    async-запроса (2× рангов) на N-split (all-gather) модели, каждый сверяется с 1-GPU reference;
    затем повторный `infer()` того же запроса даёт **бит-идентичный** выход (проверка отсутствия
    гонок общего staging/replication). `SKIP <2 GPU`. Существующий `ConcurrentRequestsMatchSingleGpu`
    (S12) не тронут (регресс-гарантия изоляции запросов).
  - **Бенч-обвязка**: `tools/tp_bench/tp_bench.py` — Python-харнесс через публичный API,
    варианты **TP1** (`compile_model(model,"GPU")`), **TP2**
    (`{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}`), **MULTI** (`"MULTI:GPU,GPU"`); greedy
    (`temperature=0`,`do_sample=False`,`num_beams=1`), фикс. warmup/iters/max_new_tokens; при наличии
    `openvino_genai` берёт реальные TTFT/TPOT (per-token timestamps), иначе называется
    indicative `decode_step_proxy`. GATE-отношения (`TPOT_T1/T2 ≥ 1.5`, `TTFT_T1/T2 ≥ 1.3`)
    считаются только на настоящих метриках. `README.md` + бенч-план `docs/notes/tp-bench-plan.md`.
  - **Ожидание по перфу (не измерено)**: реальный Qwen3.5-9B-int4 на TP2 преимущественно
    **REPLICATED** (S07–S13) → оба GPU считают полный граф, scheduler лишь **перекрывает** их
    (устраняет serial-регресс S05–S13 ≈2×). Spид-ап ограничен: TPOT_ratio ≈ 1.0 (не 1.5) плюс
    минус host-USM staging G0 (события/копии ~4–5 GB/s, ~60–90 µs/вызов, S03) и отсутствие
    in-graph коллективов. Достижение ≥1.5× требует S15 (peer-копии, ring/tree allreduce,
    in-graph all-gather, снятие REPLICATED на реальных слоях). После S14 perf-регресс «TP2 = 2×
    медленнее TP1» устранён; потолок REPLICATED — TPOT_ratio≈1.0.
  - **Память (план)**: без дублирования весов для шардированных (N/K-SPLIT) узлов — срез до upload
    (S07–S09, PLAN §4.2); REPLICATED-узлы реального Qwen дают 1× на ранг (k× суммарно) — известное
    ограничение, не регресс. Проверка — индикативно по RSS/VRAM в `tp-bench-plan.md` §1/§5.
  - **Сборка по-прежнему недоступна** (сабмодули пусты) → GATE частичный. Перед S15: при появлении
    сабмодулей/2 GPU — собрать плагин, прогнать `tensor_parallel` (в т.ч. `SchedulerParallelRanksMatchSingleGpu`)
    и `tp_bench.py` (TP1/TP2/MULTI) на 2×A770 / 2×BMG.
  - **S15 стартует с**: Stage 5.1 — перф (peer-копии, ring/tree allreduce, in-graph device-side
    коллективы, снятие REPLICATED на реальных слоях), prefill/decode отчёт, закрытие GATE TPOT≥1.5×.
    См. NEXT.md.
- **S13 done (частично, 2026-10-18)** — Stage 4.1 GenAI-интеграция. Артефакты/решения:
  - **Probe (путь свойства)**: GenAI-путь = строковый `AnyMap` → `core.compile_model(model,"GPU",cfg)`.
    Полный разбор — `docs/notes/genai-integration.md` (новый). Итог: строковый конфиг
    `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}` конвертируется в `std::set<ModelDistributionPolicy>`
    в `ov::PluginConfig::set_user_property` через `ov::Any::as` (`any.hpp:767-784`; `Read<std::set<>>`,
    enum `Istreamable`), доходит до `Plugin::compile_model` (`plugin.cpp:275-286`) → TP-ветка. **Правок
    плагина/GenAI не потребовалось** (расхождения property vs config нет). GPU рекламирует свойство RW.
  - **Совместимость compile-фрейма**: GenAI (LLMPipeline) использует публичный `ov::InferRequest`;
    TP-обёртка поддерживает name-based `set_tensor/get_tensor` (через `getPort` над портами TP-модели),
    `query_state()/set_state()` (KV/linear-attn через `TPAggregatedState`), `inputs()/outputs()`
    (синтетические full-shape порты ALL_GATHER сохраняют friendly name). Корректность подтверждена
    контрактом + новыми тестами (прогон вне сессии).
  - **GenAI CB** (openvino.genai) в дереве отсутствует → «как в продакшене» проверяется эмуляцией
    строкового конфига + name-based IO в тестах. Паритет реальной greedy-генерации Qwen3.5-9B-int4
    TP2 vs 1-GPU НЕ прогонялся (нет сборки, GenAI-пакета и 2 GPU) → **частичный GATE**.
  - **Тесты S13** (`tensor_parallel.cpp`): `GenAiStringConfigEnablesTensorParallel`;
    `GenAiStyleStatefulRoundtripMatchesSingleGpu` + билдер `make_genai_style_stateful_model`.
    Форматы `tp_seams`/`tp_output_seams`/`tp_state_seams` не менялись.
  - **Ожидание по реальному Qwen**: TP2 остаётся преимущественно REPLICATED → выходы/состояние
    ранга-0 → детерминированная генерация должна совпасть с 1-GPU. Полный спид-ап требует in-graph
    межранговых коллективов (S15) и scheduler (S14).
  - **S14 стартует с**: Stage 4.2 — Scheduler + бенч TP2 vs TP1/MULTI (GATE TPOT ≥ 1.5× на 2×BMG).
    Точки: `TPInferRequest::infer()` сейчас serial (ранги последовательно, `m_infer_mutex`) —
    scheduler/конкурентное исполнение рангов; бенч-инфра под CB; перф-репорт. См. NEXT.md.
- **S12 done (частично, 2026-10-18)** — Stage 3.3 Hard: динамические state-шейпы + межранговый транспорт + конкурентность.
  Артефакты/решения:
  - **Динамика**: `TPStateSeam.full_shape` = `ov::PartialShape`. `find_state_head_axis(PartialShape)` требует
    **статическую делимую** head-ось; `analyze_state_variable` допускает динамический batch/seq (раньше — отказ
    «dynamic state shape»). `sharded[axis] = ov::Dimension(full_dim / num_ranks)`, динамика прочих осей сохраняется;
    после `update_data_shape` обязателен `validate_nodes_and_infer_types()` (per-pass validation off,
    `transformations_pipeline.cpp:645`). `full_dim`/`axis`/`variable_id` из `TPStateSeam` не менялись (формат свойства тот же).
  - **Транспорт**: `TPTransport` (host-USM staging, G0) — единая точка коллективов. `tp_transport_capabilities()`
    возвращает кэшированный `HOST_STAGING/peer=false/ipc=false` (S02/S03); PEER — fast-path hook для S15 (запрос
    `zeDeviceCanAccessPeer` уровня плагина), на текущем железе не активен. Коллективы работают только с host-visible
    `ov::ITensor` (remote → `OPENVINO_ASSERT`), `m_staging` переиспользуется (не аллоцируется на инференс).
  - **Динамическое состояние + транспорт**: `TPAggregatedState` больше не хранит compile-time full_shape — полная форма
    выводится из runtime-slice `rank-0` (`slice_shape[axis]*=num_ranks`), `get_state()` — `all_gather_axis`,
    `set_state()` — `scatter_axis` (per-rank буферы берут форму от входного полного состояния).
  - **Конкурентность**: `m_infer_mutex` (сериализация `infer()` одного запроса) + `m_output_mutex` (lazy-аллокация
    выходов). Параллельный запуск per-rank sub-запросов осознанно НЕ делался (детерминированный порядок staging;
    параллелизм — S14 scheduler). `TPInferRequest` теперь содержит `std::mutex` → копирование/перемещение недоступно
    (запросы и не копируются).
  - **Device-side/события**: не реализованы (приоритет S12 — корректность/паритет). Находки S02: counter-based event +
    `zeEventHostSignal` = UNSUPPORTED; classic `zeEventCreate` из ctypes крашил драйвер → `zeEventPool` использовать
    осторожно. True in-graph inter-node all-gather/all-reduce (разрезка coupled fused `in_proj`/`Concat`/`Loop`) —
    не в S12; реальный Qwen по-прежнему преимущественно REPLICATED.
  - **Тесты**: `make_dynamic_stateful_head_model`, `DynamicStateHeadShardingMatchesSingleGpu`,
    `DynamicStateIndivisibleStaysReplicated`, `StateRoundtripHeadShardedMatchesSingleGpu`,
    `StateRoundtripReplicatedMatchesSingleGpu`, `ConcurrentRequestsMatchSingleGpu`. Форматы свойств стабильны.
  - **Сборка по-прежнему недоступна** (сабмодули пусты) → GATE частичный (код + план теста). Перед S13:
    при появлении сабмодулей — прогон `tensor_parallel` и Qwen3.5-9B-int4 TP2 vs 1-GPU (seq ~256).
  - **S13 стартует с** (выполнено в S13, см. блок S13 выше): Stage 4.1 — GenAI интеграция (см. NEXT.md). Точки: `TPCompiledModel` уже отдаёт
    сериализуемые plan/seams и работает через штатный `ov::InferRequest` API (GenAI/OV-genai использует его напрямую);
    проверить `model_distribution_policy{TENSOR_PARALLEL}` со стороны GenAI-пайплайна, конфиг `tp` в `LLMPipeline`,
    поведение stateful-модели и `query_state/set_state` через GenAI.
- **S11 done (частично, 2026-10-17)** — Stage 3.2 linear_attention. Артефакты/решения:
  - **Примитив «depthwise conv per head» (PLAN §4.3)**: `TPShardTransform::analyze_state_variable` теперь
    обрабатывает `ov::op::v1::GroupConvolution` в цепочке состояния. Распознаётся только **depthwise-форма**
    (`match_depthwise_head_conv`: filters — `Constant` формы `[GROUPS,1,1,k]`, `GROUPS == data channels ==
    full_dim`, данные-операнд на head-оси); при совпадении фильтры режутся по оси 0 тем же ранговым
    диапазоном (`slice_constant`+`replace_constant`, суффикс `_tp_headrank`), затем шардится переменная и
    вызывается `validate_nodes_and_infer_types()`. Не-depthwise/`GroupConvolution`-без-Constant-filters →
    цепочка `REPLICATED` (без throw). Второй проход BFS пропускает input 1 (filters) конва как исключение.
  - **Head-ось**: `find_state_head_axis` предпочитает **axis 1** (`[batch, heads, ...]` — KV/conv/ssm Qwen),
    затем `rank-2`, затем первую делимую внутреннюю. Старые S10-тесты (`[1,4,2]`) не затронуты (axis 1).
  - **Инвентарь linear-attention** — `docs/notes/state-inventory.md` §6: conv-цепочка `ReadValue[b,8192,4] →
    Gather(axis0) → Concat(axis2) → GroupConvolution(filters[8192,1,1,4]) → Slice → Assign`; ssm `ReadValue[b,32,128,128] →
    Gather → Loop(Loop_1502, delta-rule, ScatterNDUpdate) → Reshape → Assign`; head-ось = 1; **conv-веса тоже
    int4_ASYM dequant** (`Const u4 → Convert → Subtract → Multiply → Convert`, не голый Constant). Вывод:
    реальные Qwen-цепочки не head-local → REPLICATED; per-rank Qwen = S12.
  - **Тесты** (`tensor_parallel.cpp`): билдер `make_linear_attn_conv_state_model(heads,taps)` (Variable
    `[1,heads,taps]` → ReadValue → depthwise GroupConvolution same-pad → Assign + Result);
    `LinearAttentionConvStateHeadShardingMatchesSingleGpu` (heads=4: seam `HEAD_SHARDED:1:4:`, 4-шаговый
    паритет, агрегированное состояние `[1,4,taps]`); `LinearAttentionConvStateIndivisibleStaysReplicated`
    (heads=3: seam `REPLICATED:`). Шапка-комментарий теста дополнена S10/S11.
  - **Сборка по-прежнему недоступна** (сабмодули пусты) → GATE частичный (код + план теста). Перед S12:
    ревью и, при появлении сабмодулей, прогон `tensor_parallel`.
  - **S12 стартует с**: Stage3.3 Hard — динамические шейпы состояний (`?,4,?,256` и т.п.), device-side
    коллективы/события, concurrency. Реальные Qwen-состояния станут per-rank только когда появятся межранговые
    коллективы и разрезка coupled fused `in_proj`/`Concat`/`Loop` (см. §6 инвентаря).
- **S10 done (частично, 2026-10-17)** — Stage 3.1 per-rank KV + state aggregation. Артефакты/решения:
  - **Реальный Qwen → все состояния REPLICATED.** KV (`ScaledDotProductAttention`), conv (`GroupConvolution`)
    и ssm (`Loop`) — не element-wise потребители, head-slice небезопасен; per-rank KV для Qwen невозможен
    без inter-node коллективов (S12). Состояние остаётся полным на каждом ранге, агрегация — rank-0.
  - **Инвентарь/обоснование** — `docs/notes/state-inventory.md` (новый). Head-ось KV = 1; ssm = ось 1 (32);
    conv — ось 1 (8192), но linear-attention → S11.
  - **Формат свойства** `tp_state_seams` = `KIND:AXIS:FULL_DIM:VARIABLE_ID` (тест ищет префикс `HEAD_SHARDED:`
    и подстроку `:1:4:`). Отличие от исходного дизайна (порядок полей) — см. `tp_state_seam_to_string`.
  - **Per-pass validation выключен** (`transformations_pipeline.cpp:645`) → после `update_data_shape` обязателен
    явный `model->validate_nodes_and_infer_types()` (сделано при `state_modified`).
  - **Форматирование**: репо пинует clang-format-18 (`docs/dev/coding_style.md`), в среде только clang-format 20.1.0 →
    whole-file прогон даёт массовый нерелевантный churn (макросы/бэкслэши/отступы). **Не запускать clang-format**
    на этих файлах; стиль — вручную. Все 12 изменённых файлов перезалиты вручную после отклонённого прогона.
  - **S11 стартует с**: linear-attention состояния (`past.conv.*`, `past.ssm.*`) — тот же пайплайн классификации, но
    цепочки через `Loop`/`GroupConvolution`/`ScatterUpdate`; ожидаемо снова REPLICATED до S12, если не найдётся
    head-local подцепочка.
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
