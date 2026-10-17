# СЕССИЯ S10 — Stage 3.1 per-rank KV + GQA-головы + агрегация состояний · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §2 (численные инварианты:
     GQA 16 q-голов / 4 kv-головы / head_dim 256; головы и K-оси проекций кратны числу рангов),
     §4.2 (compile-time, срез до `propagate_constants`), §4.3 (шардинг классов слоёв: q/k/v/o, MLP,
     attention-головы), §5 (строка S10), §6 (тест-матрица), §7.
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая», блок **S09→S10**
     (int4-срез сделан на S09; inter-node all-gather и внутренний residual НЕ реализованы — актуальные
     ограничения для S10) и блоки S06/S07/S08.
  3) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — паттерн TP (что режется/что реплицируется; per-rank state).
  4) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3 (pass до `propagate_constants`), §о stateful/профиле.
  5) C:\openvino-tp\docs\notes\int4-asym-weights.md — форма dequant-подграфа и правила среза (S09).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S10 (таблица PLAN.md §5). Следующие этапы
  (S11 linear_attention, S12 device-side коллективы/события, S13 GenAI, S14+ бенч/перф/стабилизация)
  ЗАПРЕЩЕНЫ. Остановка на GATE/блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC NA). **G0 = host-USM
  staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Device-side швы — S12.
- S04 done: `TPCoordinator`/TP-ветка `compile_model`/свойства (`tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy`).
- S05 done: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы только rank0;
  `query_state` сейчас — ТОЛЬКО rank-0, без агрегации).
- S06 done: `TPShardTransform` классифицирует `MatMul`/`FullyConnected(Compressed)`:
  нет константных весов → `REPLICA`; выход фида в `ov::op::v1::Add` → `K_SPLIT`(`ALL_REDUCE_SUM`);
  иначе → `N_SPLIT`(`ALL_GATHER`). `tp_seams` = `KIND:COLLECTIVE:TYPE:NAME`.
- S07 done: реальный N-split (срез Constant по N-оси + bias input 2) + host all-gather-эквивалент
  (`TPOutputSeam` ALL_GATHER, синтетический полный порт, конкатенация регионов). Rank прокинут в пасс через
  internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks` → `register_pass<TPShardTransform>(rank, num_ranks)`.
- S08 done: K-split + all-reduce(sum) — K-срез весов и активации (`ov::op::v8::Slice`, `axis=-1`);
  host-side subtractive reducer `TPInferRequest::assemble_reduce_sum` (fp32-сумма частичных,
  `- (num_ranks-1)*term` для replicated bias/residual `TPReplicatedAdd` CONSTANT/INPUT);
  `TPOutputSeam.collective`+`replicated_adds`; RO-property `ov::intel_gpu::tp_output_seams`
  (формат `COLLECTIVE:producer:split_axis:shard_dim_size:repl_adds`).
- **S09 done (главный контекст для S10):** реализован срез int4_ASYM:
  - `match_quantized_weight` распознаёт dequant-подграф на операнде 1 GEMM
    (`weight u4 [..,N,G,gs] -> Convert -> Subtract(zp) -> Multiply(scale) -> Reshape([..,N,K]) -> Convert`),
    `has_constant_weights` теперь принимает и bare-Constant, и dequant-подграф.
  - N-split: weight+scale+zp по N-оси целыми oc-чанками + обновление target `Reshape` (`n_out→count`).
    K-split: weight+scale+zp по group-оси целыми группами (`G % num_ranks == 0`, K кратно `gs*ranks`) +
    `Reshape` (`K→count_g*gs`); активация — как в S08. `zp` per-tensor (1 элемент) НЕ режется.
  - Sub-byte выравнивание: `slice_is_byte_aligned` (row/off/slab_bits % 8). Невыровненно/неделимо →
    REPLICA без throw. `is_sliceable_type` допускает любую статическую ширину > 0.
  - `scan_nsplit_consumers` (BFS по elementwise) заменил `consumers_are_slice_safe`.
  - Тесты: `Int4NsplitParityMatchesSingleGpu`, `Int4KsplitParityMatchesSingleGpu`,
    `Int4KsplitIndivisibleFallsBackToReplica`.

## Задача S10 (полный объём этой сессии)
Stage 3.1 — **per-rank KV + GQA-головы + агрегация состояний**. Учить TP-путь работать со stateful
(attention KV) моделями: KV-кэш шардируется по головам (per-rank), а `TPInferRequest` агрегирует
per-rank состояние. Цель GATE — **greedy-паритет 32 токенов** против 1-GPU на Qwen3.5-9B-int4.

Qwen3.5 LM: 32 слоя = 24 × linear_attention + 8 × full_attention; MLP 4096→12288→4096; GQA
16 q-голов / 4 kv-головы / head_dim 256. Число kv-голов (4) и q-голов (16) делится на 2 (и 4) ранга;
head_dim=256.

1. **Инвентаризация state в модели.**
   - Определить, как KV-кэш представлен в графе до/после `transformations_pipeline` (Stateful ops,
     `ReadValue`/`Assign`, или переменные `ov::Model::get_variables()`/`get_sinks()`); проверить на
     `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml` (только чтение; `.bin` не коммитить).
   - Определить, как linear_attention-слои хранят состояние (свёрточное/рекуррентное состояние) —
     это НЕ KV-кэш; зафиксировать его форму и пометить как «S11», если шардинг нетривиален; в S10
     линейное-внимание состояние можно оставить реплицированным/агрегируемым как есть, если паритет
     достигается.
   - Задокументировать фактические формы (число голов × head_dim, оси head-размерности).

2. **Per-rank шардинг KV/голов.**
   - Шардировать KV-головы по рангам: каждая голова целиком на одном ранге (число kv-голов кратно
     числу рангов). Согласовать с N-split/K-split проекций (q/k/v/o) там, где это уже режется S07/S09;
     если цепочка qkv→reshape→rope→sdpa не выдерживает конкатенацию срезов (S07 §ограничение), выбрать
     и обосновать: репликация либо шардинг по головам.
   - Учесть соотношение GQA: q-головы = kv-головы × ratio; при N-split q-проекции срез по головам должен
     сохранять пары (q-голова → соответствующая kv-голова) согласованно между рангами.
   - НЕ реализовывать device-side коллективы/события (S12): сборка состояния между рангами — host-side
     (как S07/S08).

3. **Агрегация состояния в `TPInferRequest`.**
   - `query_state` сейчас возвращает состояние только rank-0. Реализовать сбор/агрегацию per-rank состояния
     (`get_variables`/`get_state`/`query_state`) в согласованной форме (конкатенация head-размерности при
     шардинге по головам; для реплицированного состояния — копия с rank-0).
   - Отрефакторить существующие `assemble_reduce_sum`/all-gather-конкатенацию, если переиспользуемо;
     сохранить контракт «выходы/состояние только со стороны rank-0».
   - Обработать сброс/общий доступ к состоянию (`reset_state`/`set_state`), если публичный контракт требует.

4. **Не регрессировать S04–S09.**
   - Паритеты S05/S07/S08/S09, S04-негативы, S06-seams, TP-off — зелёные по коду/логике.
   - `tp_seams`/`tp_output_seams` формат стабилен (или расширяй согласованно, обновляя тесты).
   - Fallback при неделимости/небезопасном потребителе/невыровненности → репликация без throw (как S07–S09),
     с записью причины в debug-лог.

5. **GATE S10.**
   - Главный GATE: **Qwen3.5-9B-int4 TP2 greedy 32-токена паритет vs 1-GPU** (совпадение argmax на 32
     шагах декода). Если модель/сборка недоступны — **частичный GATE**: код + план теста + статическая
     проверка ревью; честная пометка в STATE. Модель: `models/Qwen3.5-9B-int4-ov/`.
   - Минимальный функциональный тест (обязателен, если общая сборка недоступна): stateful-мини-модель с
     KV/головами, TP2 vs 1-GPU паритет на нескольких шагах, проверка per-rank state и агрегации. SKIP <2 GPU.
   - Негатив: число kv-голов не кратно числу рангов → репликация (без throw) и паритет.

## GATE S10 (критерии завершения)
- per-rank KV/головы шардируются (или обоснованно реплицируются) + `TPInferRequest` агрегирует состояние.
- Паритет Qwen3.5-9B-int4 TP2 greedy 32 токена vs 1-GPU (или частичный GATE: код + план теста).
- S04–S09 не сломаны; `TPPlan`/`tp_seams`/`tp_output_seams` стабильны; TP-off == upstream.
- STATE.md: S10 done/блокер + артефакты + решения (форма state, шардинг голов/GQA, агрегация) + заметки → S11.
- NEXT.md — полный самодостаточный промпт S11 (linear_attention + агрегация состояния).
- Commit (сообщение: "S10: per-rank KV + GQA head sharding + state aggregation") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S11+ (linear_attention как отдельная задача, device-side коллективы/
  события как прод-путь, GenAI, бенч). S10 = только per-rank KV/головы + агрегация state.
- Не коммитить `.bin`/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать
  блокер в STATE и считать GATE частичным (код + план теста готовы).
