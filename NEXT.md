# СЕССИЯ S08 — Stage 2.3 K-split + allreduce (fp32-аккум, bias в reducer, швы-события) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §4.2 (compile-time, место
     TP-пасса, срез до `propagate_constants`), §4.3 (K-split O-proj/down-proj + allreduce(sum), bias в
     reducer), §4.4 (runtime, швы-события), §5 (строка S08), §7.4 (fp16-суммы → fp32-аккум).
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая» блок **S07→S08**
     (что уже сделано: N-split срез, TPOutputSeam/assembly, rank-plumbing) и S04–S06.
  3) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — §5–§6 (allgather = конкатенация; CPU K-split/allreduce
     ОТСУТСТВУЕТ — проектируем с нуля).
  4) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3–§4 (pass до propagate_constants; очереди/события).
  5) C:\openvino-tp\docs\notes\l0-spike.md — §8 (G0 = host-USM staging; host-sync floor ~60–90 µs).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S08 (таблица PLAN.md §5). Следующие этапы
  (S09+ — int4_ASYM, per-rank KV, device-side коллективы/события как прод-путь) ЗАПРЕЩЕНЫ. Остановка на
  GATE/блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC NA). **G0 = host-USM
  staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Device-side швы — S12; на S08 допустима host-сборка.
- S04 done (частично): `TPCoordinator`/TP-ветка `compile_model`/свойства (`tp_ranks`,
  `enable_tensor_parallel`, `model_distribution_policy`).
- S05 done (частично): `TPInferRequest` (k per-rank sub-запросов, репликация входов, последовательный
  infer, выходы через `TPCompiledModel`).
- S06 done (частично): `TPShardTransform` детектит `MatMul`/`FullyConnected(Compressed)` и классифицирует:
  нет константных весов → `REPLICA`; выход фида в `ov::op::v1::Add` → `K_SPLIT`(`ALL_REDUCE_SUM`); иначе
  → `N_SPLIT`(`ALL_GATHER`). Пишет `std::vector<TPSeam>` в rt_info `intel_gpu_tp_seams`.
- **S07 done (частично):** реальный N-split + host all-gather-эквивалент.
  - Rank прокинут в пасс (Вариант A): internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks`
    (`internal_properties.hpp`+`options.inl`), штампуются `TPCoordinator`
    `set_property` в per-rank config; `TransformationsPipeline::apply` →
    `register_pass<TPShardTransform>(config.get_tp_rank(), config.get_tp_num_ranks())`.
  - `TPShardTransform` для `N_SPLIT` с константными весами режет Constant по N-оси (`slice_constant`,
    `offset=rank*N/ranks`; ось по `transpose_b`), режет bias (input 2), заполняет
    `TPSeam.split_axis=-1`/`shard_dim_size=N`, возвращает `true`. Sub-byte (int4) — throw (S09).
    Fallback при неделимости/небезопасном потребителе → `REPLICA` без throw.
  - **K_SPLIT/REPLICA узлы сейчас НЕ режутся** — копируются целиком на каждый ранг
    (это то, что делает S08).
  - All-gather outputs: `TPOutputSeam` (rt_info `intel_gpu_tp_output_seams` → `TPPlan.output_seams`),
    `TPCompiledModel::build_output_ports()` публикует полный синтетический порт,
    `TPInferRequest::assemble_outputs()` конкатенирует регионы rank-выходов (host, без арифметики).
  - Тесты S07: `NsplitMlpParityMatchesSingleGpu`, `NsplitNotDivisibleFallsBackToReplica`.

## Задача S08 (полный объём этой сессии)
Stage 2.3 — **K-split + allreduce**: для узлов, помеченных `K_SPLIT` (residual-проекции: O-proj,
MLP down-proj, linear-attention out-proj), срезать константные веса по оси сокращения (K) так, чтобы на
устройство каждого ранга уезжал только его K-слайс (на OV-уровне, ДО `propagate_constants`/upload).
Результат каждого ранга — частичная GEMM-сумма; собрать полный результат суммированием частичных
(все-редеью(sum), fp32-аккумуляция). K-split/allreduce должен работать вместе с уже реализованным N-split.

1. **K-split срез весов (K-ось).**
   - В `TPShardTransform` для `kind == K_SPLIT` и константных весов: срезать веса по K-оси.
     Для `MatMul` с `transpose_b=false` веса `[K, N]` → срез по axis 0 (`[k_start:k_end, :]`); при
     `transpose_b=true` веса `[N, K]` → срез по axis 1. Для `op::FullyConnected` — учесть `transpose_b`.
   - **Вход активации тоже надо срезать по K** (это отличие от N-split!): A-операнд (input 0) — не
     константа, поэтому K-срез входа делается... варианты: (а) на S08 допускается оставить A полным, но
     тогда нужен K-split только по весам и суммирование частичных N-полных выходов — это корректно
     ТОЛЬКО если каждый ранг считает `A_full @ W[:, k_r]`, что даёт частичную сумму по k_r (нужно
     суммировать по рангам) — но тогда каждый ранг держит полный A и полный выход N, экономя лишь
     веса по K. Уточни/зафиксируй решение и его численный инвариант в STATE. Приоритет — корректность
     паритета; K-срез активаций (если требуется) — в этом же пассе через `StridedSlice` по последней
     оси A с границами кратными group_size (для int4 — S09).
   - Численные инварианты (PLAN §2): K-границы кратны `group_size` (=128 для int4); bias НЕ резать
     (добавляется один раз в reducer).
   - **scales/zp** не трогать (int4 — S09), но не сломать.
   - Не регрессировать N-split (S07): N и K срезы должны корректно сосуществовать в одном пассе.

2. **Allreduce(sum) — сборка полного результата.**
   - Host-side достаточно на S08 (device-side staging/события — S12, см. G0): после инференса все ранги
     отдают свои частичные результаты; reducer суммирует их в fp32 и кладёт в выходной буфер rank-0.
   - **bias добавляется ОДИН раз** (после суммирования), а не в каждом ранге.
   - **fp32-аккумуляция** обязательна (fp16-суммы запрещены, PLAN §7.4): читать частичные в fp32,
     суммировать, только затем кастовать к выходному типу.
   - Контракт выходов S05 сохранить: выходы только на rank-0, полный тензор.
   - Расширить `TPPlan`/rt_info новым механизмом (аналог `TPOutputSeam`, но с `TPCollectiveKind::
     ALL_REDUCE_SUM` и списком рангов-участников), не ломая `output_seams` (N-split/all-gather).
   - `TPInferRequest::assemble_outputs()` уже владеет выходными буферами — добавить ветку allreduce:
     для output-seam с `ALL_REDUCE_SUM` суммировать per-rank выходы (fp32) в dst rank-0.
   - Определение всех выходов, требующих allreduce: model-output, достижимый из `K_SPLIT`-узла (через
     elementwise/none). Возможно понадобится объединить N- и K-выходы в один список с полем коллектива.

3. **Швы-события (минимально).**
   - На S08 достаточно host-синхронной сборки (S07-стиль). Реальные device-side швы (IPC-события,
     staging-буферы, `SubMemoryManager`-аналог) — S12. Если решишь ввести заготовку событий — не выходи
     за рамки этапа; журнал решений — в STATE.

4. **Не ломать S04/S05/S06/S07.**
   - Паритеты `TensorParallelInferMatchesSingleGpu` (S05), `NsplitMlpParityMatchesSingleGpu` (S07),
     S04-негативы, S06-seams-тесты, TP-off регресс — должны оставаться зелёными по коду/логике.
   - `tp_seams` формат стабилен (или расширяй строку согласованно, обновив тесты).

5. **Тесты/GATE** (`src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp` или соседний файл
   того же каталога — ROOT-glob подхватит без правки CMake):
   - **Главный GATE S08:** residual-проекция (Parameter → MatMul(weights) → Add(residual) → Result, где
     MatMul классифицируется как `K_SPLIT`) собирается в TP2; частичные результаты рангов суммируются
     (allreduce) и полный выход совпадает с 1-GPU референсом с `cos > 0.999` (и abs/rel). SKIP <2 GPU.
     Проверить, что bias (если есть) добавлен ровно один раз.
   - Комбинированный тест (опц.): N-split → elementwise → K-split (residual) — если срез активаций не
     реализован, честно ограничить сценарий и задокументировать.
   - TP-off регресс: без TP веса не режутся, `tp_seams == {}`.
   - Опц.: K-split на Qwen (24 linear + 8 full attention) — если модель и сборка доступны; иначе частичный GATE.
   - Негатив: K, не делящийся на число рангов/на group_size → fallback `REPLICA` (как в S07) либо
     `OPENVINO_THROW` — выбери и задокументируй.

6. **Регистрация**: новые `.cpp` под `src/plugin/**` подхватывает `file(GLOB_RECURSE PLUGIN_SOURCES
   src/plugin/*.cpp ...)` (`src/plugins/intel_gpu/CMakeLists.txt:147`); тест — ROOT-glob. Правок CMake
   быть не должно.

## GATE S08 (критерии завершения)
- K-split реализован: константные веса `K_SPLIT`-узлов срезаны по K-оси ДО upload; на устройство ранга
  едет только его K-слайс.
- Allreduce(sum) с fp32-аккумуляцией даёт корректный полный результат; bias добавлен один раз.
- GATE-тест: residual-проекция паритет TP2 vs 1-GPU `cos > 0.999` (при недоступной сборке — код+план
  теста, честная пометка «частично»).
- N-split (S07) не сломан; `TPPlan`/`tp_seams` стабильны; TP-off == upstream.
- STATE.md: S08 done/блокер + артефакты + решения (K-cut активации, механизм allreduce, bias) + заметки → S09.
- NEXT.md — полный самодостаточный промпт S09 (int4_ASYM: веса+scales/zp до group-границ, K кратно 128;
  GATE — Qwen3.5-9B-int4 TP2 prefill паритет).
- Commit (сообщение: "S08: K-split + allreduce (fp32 accumulation, bias in reducer)") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S09+ (int4_ASYM group-срез, per-rank KV, device-side коллективы/
  события как прод-путь, linear_attention). S08 = только K-split + allreduce.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать
  блокер в STATE и считать GATE частичным (код + план теста готовы).
