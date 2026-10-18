# TP-GPU: план выноса в upstream (серия PR, S16)

Канон: PLAN.md §8. Вся работа выполнена **локально** в `feature/gpu-tensor-parallel` (remote не трогали).
Этот документ описывает, как разложить накопленные изменения на независимо ревьюируемую серию PR
в `openvino` (upstream), в каком порядке и что в каждый PR входит.

Принципы серии:
- **Один связный слой на PR**; каждый PR собирается и проходит `*TensorParallel*`-тесты своего слоя.
- **Не менять non-TP поведение**: TP включается только при `model_distribution_policy{TENSOR_PARALLEL}`
  или `enable_tensor_parallel`; обычный путь компиляции/кэша байт-в-байт прежний.
- **Фича-флаг/эксперимент до готовности перфу**: TP помечен экспериментальным; свойство `enable_tensor_parallel`
  — internal. Публичный вход — `ov::hint::model_distribution_policy`.
- Никаких чужеродных рефакторингов вне TP-кода.

## PR 1 — TP-скелет: координатор, compiled model, entry-point

**Роль**: каркас без шардинга (S04–S05). Ранги = реплики; входы реплицируются, выходы с ранга 0.

Содержимое:
- `compile_model_tensor_parallel` + ветка в `Plugin::compile_model`; `TPCoordinator`.
- `TPCompiledModel`, `TPInferRequest` (serial-исполнение), `TPPlan` (ranks/device_ids).
- internal-свойства: `enable_tensor_parallel`, `tp_ranks`; реклама `model_distribution_policy` (RW).
- Тесты: `CompilesMiniModelOnTwoGpus`, `InternalFlagAlsoEnablesTensorParallel`, `DefaultCompileHasNoRanks`,
  `RejectsExplicitDeviceId`, `RejectsPipelineParallel`, `TensorParallelInferMatchesSingleGpu`.
- `docs/notes/{gpu-plugin-arch,cpu-tp-pattern}.md` (разведка).

## PR 2 — Анализ/классификация швов (`TPShardTransform`, skeleton)

**Роль**: identity-пас, который детерминированно классифицирует `MatMul`/`FullyConnected(Compressed)`
(N_SPLIT/K_SPLIT/REPLICA) и пишет `TPSeam` в rt_info. Слайсинга ещё нет.

Содержимое:
- Пас + регистрация в `TransformationsPipeline::apply` только при TP.
- `TPPlan.seams`, RO-свойство `tp_seams` (`KIND:COLLECTIVE:TYPE:NAME`).
- Тесты: `ShardTransformDetectsAndReportsSeams`, `ShardTransformClassifiesResidualProjectionAsKSplit`,
  `DefaultCompileHasNoSeams`.

## PR 3 — N-split: срез весов + all-gather

**Роль**: первый реальный шардинг (S07).

Содержимое:
- internal `tp_rank`/`tp_num_ranks`; штамп per-rank в `ExecutionConfig`.
- Срез Constant по N-оси (`transpose_b`), срез bias; fallback REPLICA при неделимости/не-elementwise.
- `TPOutputSeam` (all-gather), синтетический full-shape порт ранга 0, сборка concat.
- Тесты: `NsplitMlpParityMatchesSingleGpu`, `NsplitNotDivisibleFallsBackToReplica`.

## PR 4 — K-split: срез весов+активаций + all-reduce(sum)

**Роль**: сокращение по K с частичными суммами и all-reduce (S08).

Содержимое:
- Срез Constant по K-оси (`find_weight_k_axis`), срез A-операнда через `v8::Slice`.
- `scan_k_reduction` → `TPReplicatedAdd` (CONSTANT/INPUT), `assemble_reduce_sum` (`acc -= (k-1)*term`), fp32.
- RO-свойство `tp_output_seams`.
- Тесты: `KsplitResidualParityMatchesSingleGpu`, `KsplitNotDivisibleFallsBackToReplica`.

## PR 5 — int4_ASYM: срез dequant-подграфа

**Роль**: поддержка сгруппированного `u4` (S09).

Содержимое:
- `match_quantized_weight`; срез weight/scale/zp (N-чанки / group-ось) + обновление `Reshape`;
  `slice_is_byte_aligned` (`row_bits % 8`) → иначе REPLICA.
- Тесты: `Int4NsplitParityMatchesSingleGpu`, `Int4KsplitParityMatchesSingleGpu`,
  `Int4KsplitIndivisibleFallsBackToReplica`.
- `docs/notes/int4-asym-weights.md`.

## PR 6 — Состояния: per-rank KV/head-sharding + агрегация

**Роль**: `query_state()/set_state()` c per-rank распределением (S10–S12).

Содержимое:
- `TPStateSeam` (`HEAD_SHARDED`/`REPLICATED`), `analyze_state_variable`, `find_state_head_axis`,
  срез depthwise-conv фильтров; динамические PartialShape-состояния.
- `TPAggregatedState`; RO-свойство `tp_state_seams`.
- `TPTransport` (host-USM staging) — вынести в этом PR (нужен для агрегации).
- Тесты: state-head/replicated/dynamic/linear-attn + `StateRoundtrip*`.
- `docs/notes/state-inventory.md`, `l0-spike*.md`.

## PR 7 — Scheduler: параллельное исполнение рангов

**Роль**: снять serial-регресс (S14). `m_rank_executor` (stream на ранг), `schedule_ranks()`.

Содержимое:
- `tp_infer_request.{hpp,cpp}` + создание пула в `TPCompiledModel`.
- Тесты: `SchedulerParallelRanksMatchSingleGpu`, `ConcurrentRequestsMatchSingleGpu`.

## PR 8 — Перф-коллективы: peer fast-path + ring all-reduce

**Роль**: ускорение транспорта (S15).

Содержимое:
- `zeDeviceCanAccessPeer` weak-symbol, `gpu_devices_peer_accessible`, `PeerRegionCopy`,
  `set_peer_region_copy`; `tp_transport_capabilities(k)`.
- Ring reduce-scatter+all-gather `ring_all_reduce_fp32` + `last_all_reduce_stats()`.
- Тест `KsplitRingAllReduceDeterministicAcrossRuns`.
- `docs/notes/s15-perf-report.md`.

## PR 9 — Per-rank blob-cache + стабилизация

**Роль**: стабильность кэша и финальные доки (S16).

Содержимое:
- `TPCompiledModel::export_model` (конверт `"TPSB"` + план + per-rank блобы);
  `Plugin::import_model_tensor_parallel`, `import_single_device_model` (helper), маршрутизация по TP-политике.
- `save_tp_plan`/`load_tp_plan`.
- Тесты: `S16ExportImport*` (roundtrip parity, K-split, stateful, guard-тесты).
- `docs/notes/{tp-artifacts,upstream-series,s16-perf-matrix}.md`.

## Что НЕ входит в серию (tracked как follow-up)

- **In-graph межранговые коллективы** (all-gather/all-reduce внутри графа плагина) — снимает
  REPLICATED-ограничение на реальных Qwen-слоях → главный рычаг к `TPOT ≥ 1.5×`.
- **Device-side события** и peer-инициализация под конкретное железо (BMG cross-device access).
- **Zero-copy rank-import** (mmap per-rank блоба вместо пересборки из памяти).
- **Pipeline-parallel** (сейчас явно отвергается).

## Порядок ревью/мержа

PR 1→9 — **линейно зависимы** (каждый использует предыдущий). Допустимые точки «мягкой» параллельности:
PR 5 (int4) и PR 7 (scheduler) можно ревьюить одновременно с PR 4/6 после мержа PR 3. PR 9 — последним
(стабилизация после всех слоёв). Каждый PR: TP-тесты своего слоя зелёные + обычный GPU-прогон не регрессировал.
