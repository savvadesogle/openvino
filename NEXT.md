# СЕССИЯ S09 — Stage 2.4 int4_ASYM: срез весов+scales/zp по group-границам (K кратно 128) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §2 (численные инварианты
     шардинга: K-границы кратны `group_size=128`; N-split — целыми oc-группами; `scales/zp` режутся
     вместе со слайсами весов), §4.2 (compile-time, срез до `propagate_constants`), §4.3 (шардинг классов
     слоёв), §5 (строка S09), §6 (тест-матрица), §7.
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая» блок **S08→S09**
     (K-split/allreduce сделаны на S08: срез весов+активации, subtractive reducer, `TPOutputSeam.collective`
     + `replicated_adds`, `tp_output_seams`; sub-byte веса сейчас реплицируются — это задача S09) и блоки
     S06/S07.
  3) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — §5–§6 (CPU FC режет `weights`/`bias`/`scales`/`zp`
     одним способом; `zp` с 1 элементом НЕ режется; CPU `split_parts` даёт неравные хвосты — для int4
     НЕ годится, K кратно 128).
  4) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3 (pass до `propagate_constants`).
  5) C:\openvino-tp\docs\notes\l0-spike.md — §8 (G0 = host-USM staging).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S09 (таблица PLAN.md §5). Следующие этапы
  (S10+ — per-rank KV, linear_attention, device-side коллективы/события, GenAI) ЗАПРЕЩЕНЫ. Остановка на
  GATE/блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC NA). **G0 = host-USM
  staging** (dev→pinned host→dev, ~4–5 GB/s e2e). Device-side швы — S12.
- S04 done: `TPCoordinator`/TP-ветка `compile_model`/свойства (`tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy`).
- S05 done: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы только rank0).
- S06 done: `TPShardTransform` классифицирует `MatMul`/`FullyConnected(Compressed)`: нет константных
  весов → `REPLICA`; выход фида в `ov::op::v1::Add` → `K_SPLIT`(`ALL_REDUCE_SUM`); иначе
  → `N_SPLIT`(`ALL_GATHER`). `tp_seams` = `KIND:COLLECTIVE:TYPE:NAME`.
- S07 done: реальный N-split (срез Constant по N-оси, режет bias input 2) + host all-gather-эквивалент
  (`TPOutputSeam` ALL_GATHER, синтетический полный порт, конкатенация регионов). Rank прокинут в пасс через
  internal-опции `ov::intel_gpu::tp_rank`/`tp_num_ranks` → `register_pass<TPShardTransform>(rank, num_ranks)`.
- **S08 done (главный контекст для S09):** реализован K-split + all-reduce(sum):
  - K-срез весов (`find_weight_k_axis`: 2D; `transpose_b=false`→ось 0 `[K,N]`, `true`→ось 1 `[N,K]`) и
    K-срез активации (`ov::op::v8::Slice`, `axis=-1`, `start=offset`, `stop=offset+count`).
  - `slice_constant`/`is_sliceable_type` **явно отклоняют sub-byte** (`bitwidth >= 8`): int4-веса на S08
    НЕ режутся, узел просто реплицируется (не падает). **Это и есть задача S09.**
  - Allreduce — host-side subtractive reducer (`TPInferRequest::assemble_reduce_sum`): fp32-сумма частичных,
    `- (num_ranks-1)*term` для replicated bias/residual (`TPReplicatedAdd`: CONSTANT/INPUT).
  - `TPOutputSeam.collective` (ALL_GATHER/ALL_REDUCE_SUM) + `TPOutputSeam.replicated_adds`; новый RO-property
    `ov::intel_gpu::tp_output_seams` (формат `COLLECTIVE:producer:split_axis:shard_dim_size:repl_adds`).
  - `tp_seams`-формат НЕ менялся; не-срезанные K-узлы сохраняют классификацию (стабильность тестов S06).
  - Тесты S08: `KsplitResidualParityMatchesSingleGpu`, `KsplitNotDivisibleFallsBackToReplica`.

## Задача S09 (полный объём этой сессии)
Stage 2.4 — **int4_ASYM**: научить `TPShardTransform` резать **квантованные** веса вместе с их
`scales`/`zero_points` (и `bias`, где применимо) так, чтобы и N-split, и K-split работали на
Qwen3.5-9B-int4-ov (INT4_ASYM, group_size=128, ratio=1.0), сохраняя group-границы:
- **N-split int4**: срез по out-channels (N) целыми oc-группами (границы кратны group_size по N).
- **K-split int4**: срез по K кратно `group_size=128` (K-границы совпадают с group-границами, чтобы
  каждая K-группа целиком уехала на один ранг).

Qwen3.5 LM: 32 слоя = 24 × linear_attention + 8 × full_attention; MLP 4096→12288→4096; K-оси проекций
(12288, 4096) кратны 128; GQA 16 q-голов / 4 kv-головы / head_dim 256.

1. **Определить представление квантованных весов в графе OV (до `propagate_constants`).**
   - Найти, как INT4_ASYM веса приходят в OV-граф: `FullyConnectedCompressed`
     (`src/plugins/intel_gpu/include/intel_gpu/op/fully_connected_compressed.hpp`) + `MarkDequantization`,
     либо `Convert`/`Subtract`/`Multiply`/`MatMul` (dequant-подграф `(x - zp) * scale`), либо `Constant`
     с типом `u4/i4` + отдельные `scales`/`zero_points` тензоры. Проверить на реальной модели
     `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml` (только чтение; `.bin` не коммитить).
   - Классификация S06 (`has_constant_weights`, K/N_SPLIT) должна срабатывать и на compressed/dequant
     форме (сейчас `FullyConnectedCompressed` — подкласс `FullyConnected`, но веса могут быть не «голым»
     Constant; проверить, что `node->input_value(1)` реально Constant u4/i4, или это dequant-подграф —
     тогда нужен отдельный матчинг). Задокументировать фактические формы.

2. **Срез int4_ASYM (веса + scales + zp).**
   - Реализовать в `TPShardTransform` (там же, до `propagate_constants`). Ось/арифметика как в S07 (N) и
     S08 (K), но:
     - `slice_constant` (или новый `slice_quantized_constant`) должен поддерживать sub-byte типы: u4/i4
       пакуются по 2 на байт; срез должен быть **выровнен по байтам** и не разрывать пару nibble. Если
       границы среза кратны 2 (а group_size=128 → кратны), это безопасно; иначе — fallback/throw.
     - `scales`/`zero_points` резать **тем же** диапазоном группы, что и веса, по соответствующей оси:
       для group_size=128 обычно форма `scales` = `[.., N, K/128]` или `[.., K/128, N]` — выяснить по
       факту, резать согласованно с весами.
     - `zero_points` с единственным элементом (per-tensor) НЕ резать (как CPU, cpu-tp-pattern.md §5).
     - **bias**: для K-split НЕ резать (S08: добавляется один раз в reducer / вычитается как replicated
       term); для N-split резать (как S07). Сохранить текущее поведение.
     - Численная проверка: срез не должен менять per-rank результат при последующей сборке (all-gather/
       all-reduce) — иначе паритет ломается.
   - Если веса приходят как dequant-подграф на OV-уровне (не голый Constant), выбрать стратегию: (i) матчить
     и резать компоненты dequant-подграфа (zp/scales константы), либо (ii) опереться на `FullyConnectedCompressed`
     (если он уже несёт quantized-веса как Constant) — обосновать в STATE. НЕ реализовывать срез на cldnn-уровне.

3. **Не регрессировать S06/S07/S08.**
   - Паритеты S05/S07/S08, S04-негативы, S06-seams, TP-off — зелёные по коду/логике.
   - `tp_seams`/`tp_output_seams` формат стабилен (или расширяй согласованно, обновляя тесты).
   - Fallback при неделимости/небезопасном потребителе/sub-byte-невыравненности → репликация без throw
     (как S07/S08), с записью причины в debug-лог.

4. **GATE S09.**
   - Главный GATE: **Qwen3.5-9B-int4 TP2 prefill паритет vs 1-GPU** (логиты, greedy; `cos > 0.999` или
     совпадение argmax на ≥ N токенов). Если модель/сборка недоступны — **частичный GATE**: код + план
     теста + статическая проверка ревью; честная пометка в STATE. Модель: `models/Qwen3.5-9B-int4-ov/`
     (полная, кроме `openvino_mtp_model.*`); см. PLAN §2.
   - Минимальный функциональный тест (обязателен, если общая сборка недоступна): int4-мини-модель с
     `FullyConnectedCompressed`/dequant-подграфом (N-split и K-split), TP2 vs 1-GPU паритет `cos > 0.999`,
     проверка что `tp_seams` показывает срез, а `tp_output_seams` содержит нужный collective. SKIP <2 GPU.
   - Негатив: K не кратно `group_size` (или невыравненный sub-byte срез) → репликация (без throw) и паритет.
   - Комбинированный N→K на int4 (если реализуемо без реального all-gather между узлами) — иначе
     задокументировать ограничение (как S08).

5. **Регистрация**: новых `.cpp` под `src/plugin/**` подхватывает `file(GLOB_RECURSE PLUGIN_SOURCES
   src/plugin/*.cpp ...)` (`src/plugins/intel_gpu/CMakeLists.txt:147`); тест — ROOT-glob. Правок CMake
   быть не должно.

## GATE S09 (критерии завершения)
- int4_ASYM: веса + `scales`/`zp` (и `bias` для N-split) режутся по group-границам; K кратно 128.
- Паритет Qwen3.5-9B-int4 TP2 prefill vs 1-GPU `cos > 0.999` (или частичный GATE: код + план теста).
- N-split (S07) и K-split/allreduce (S08) не сломаны; `TPPlan`/`tp_seams`/`tp_output_seams` стабильны;
  TP-off == upstream.
- STATE.md: S09 done/блокер + артефакты + решения (форма quantized-весов, срез scales/zp, выравнивание
  sub-byte) + заметки → S10.
- NEXT.md — полный самодостаточный промпт S10 (per-rank KV + GQA-головы + агрегация состояний;
  GATE — greedy 32 токена паритет).
- Commit (сообщение: "S09: int4_ASYM group-aware slicing (weights + scales/zp, K multiple of 128)") —
  **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S10+ (per-rank KV, linear_attention, device-side коллективы/события
  как прод-путь, GenAI). S09 = только int4_ASYM group-срез.
- Не коммитить `.bin`/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать
  блокер в STATE и считать GATE частичным (код + план теста готовы).
