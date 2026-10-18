# PLAN: Tensor Parallel (TP) на GPU в OpenVINO [копия для переноса в C:\openvino-tp\PLAN.md]

Дата: 2026-10-17. Базовый upstream tip: `85720fe2` (master, 2026-10-17).

## 0. Методология сессий (канон процесса)

Каждая сессия = **ровно один этап (атом)**. Контекст не переносится между
сессиями — переносится только документированное состояние.

```
ВХОД : read PLAN.md (канон) + STATE.md (прогресс) + NEXT.md (промпт)
РАБОТА: этап по своей задаче; GATE-проверка этапа
ВЫХОД : STATE.md (done/артефакты/решения/блокеры/заметки)
        + генерация NEXT.md (полный промпт следующей сессии)
        + git commit этапа + push
```

Инварианты атомарности:
1. Сессия не выполняет два этапа, даже если успевает; остановка на GATE или
   блокере — легитимный результат (фиксируется в STATE.md).
2. `PLAN.md` — канон: в сессии НЕ редактируется. Дельты/расхождения — append
   в STATE.md; правки канона — отдельной сессией.
3. Долгие исследовательские выжимки — в `docs/notes/*.md`, не в контексте.
4. Промпт сессии самодостаточен: ссылается только на PLAN/STATE/notes.

## 1. Цель и контекст

Реализовать `ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL` для устройства
**GPU** (сегодня поддержан только CPU). Реализация — внутри GPU-плагина OpenVINO.

- Локальный репо: `C:\openvino-tp` = форк `github.com/savvadesogle/openvino`,
  upstream: `github.com/openvinotoolkit/openvino` (tip `85720fe2`, 2026-10-17).
- Ветка разработки: `feature/gpu-tensor-parallel` (создаётся в S01).
- ОС: Windows primary; железо: 2+ discrete Intel GPU (Arc/Battlemage, PCIe).
- Коллективы: Level Zero peer (если доступен — решит S03/Gate G0) с
  универсальным fallback на USM host staging.

## 2. Тест-модель (единственная): OpenVINO/Qwen3.5-9B-int4-ov

Локально: `models/Qwen3.5-9B-int4-ov/` (HF-снапшот; на 2026-10-17 отсутствуют
`openvino_mtp_model.{xml,bin}` — докачать перед S10/S11, не блокер).

Факты конфига:
- VLM `Qwen3_5ForConditionalGeneration`, активации bf16, веса INT4_ASYM
  group_size=128, ratio=1.0.
- 32 слоя: **24 × linear_attention + 8 × full_attention** (каждый 4-й слой),
  GQA: 16 q-голов / 4 kv-головы / head_dim 256; hidden 4096; MLP 12288;
  vocab 248320; MTP = 1 слой.

Правила скоупа TP:
- TP-шардится **только** `openvino_language_model.xml` (MTP-модель — тем же
  путём). Vision/merger/pos/embeddings и токенизаторы — вне TP.
- TP=2 — MVP; TP=4 — secondary (kv-головы 4 → 1/ранг; key 16 → 4; value 32 → 8).
- Численные инварианты шардинга: K-split границы кратны group_size (=128);
  N-split — целые oc-группы; scales/zp режутся вместе со слайсами весов.

## 3. Ключевые факты кода (проверены на master 85720fe2)

- API уже есть: `src/inference/include/openvino/runtime/properties.hpp:436-486`
  (`ModelDistributionPolicy::{TENSOR_PARALLEL|PIPELINE_PARALLEL}`, докстринг
  упоминает "between different GPUs").
- CPU TP — эталон паттерна:
  - `src/plugins/intel_cpu/src/compiled_model.cpp:152-177` — N подмоделей,
    `SubMemoryManager`, `MessageManager`;
  - `src/plugins/intel_cpu/src/nodes/fullyconnected.cpp:241-251` (tp_cfg),
    `:294-324` (дин. шейпы dst), `:337-446` (allgather-сборка + spin-флаги),
    `:660-740` (слайс весов/bias/scales/zp по out-channels);
  - выходы биндятся только на ранге 0; состояния агрегируются
    (`infer_request.cpp:317-328`); ранги/стримы: `cpu_streams_calculation.cpp`.
- HETERO: только `PIPELINE_PARALLEL` (`src/plugins/hetero/src/config.cpp:24-35`),
  строго последовательное исполнение (`sync_infer_request.cpp:127-132`).
- GPU-плагин: TP нет полностью. `1 CompiledModel = 1 device`
  (`remote_context.cpp:59`); `num_streams` = N клонов графа на очереди ОДНОГО
  GPU (`compiled_model.cpp:70-75`); реквест последователен
  (`network.cpp:1048-1182`); события/барьеры (`stream.cpp:11-28`); IPC-заголовки
  есть и не используются (`zex_common.h:167-205`); тайлы перечисляются
  (`ze_device_detector.cpp:16-66`); константы загружаются в
  `propagate_constants.cpp:238-280`; OV-пассы: `transformations_pipeline.cpp:555+`.
- GenAI CB уже прокидывает `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}`
  (`tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp`).
- oneCCL (GPU-коллективы через L0; NCCL-подобный C API с oneAPI 2026.0) —
  MPI-центричный, вне MVP (defer).

## 4. Дизайн

### 4.1 API и свойства
- `core.compile_model(model, "GPU", {ov::hint::model_distribution_policy(
  {TENSOR_PARALLEL})})` → TP-группа = все перечисленные Intel GPU (симметрия
  с CPU/"сокеты"). Read-only `ov::intel_gpu::tp_ranks`; внутренний
  `ov::intel_gpu::enable_tensor_parallel`; отчёт TP в `get_property`;
  понятные ошибки для несовместимых комбинаций ("GPU.0" + TP и т.п.).

### 4.2 Compile-time
- `Plugin::compile_model`: TP-ветка — `TPCoordinator`:
  1) перечисление GPU (m_device_map), 2) k per-rank RemoteContext/engine,
  3) клон модели на ранг + OV-пасс `TPShardTransform` (место: рядом с
  MoE/SDPA пассами `transformations_pipeline.cpp:555+`, до фьюжнов),
  4) сборка `TPPlan` (швы: коллектив, тип, буферы, события).
- Срез констант — **до** `propagate_constants` (upload): на устройство едет
  только слайс ранга; дублирования весов нет.

### 4.3 Шардирование классов слоёв (по Qwen3.5)
- `full_attention`: QKV — N-split по головам (allgather-эквивалент: сборка в
  полный буфер без арифметики); SDPA исполняется полным окном на головы
  ранга; O-proj — K-split + allreduce(sum). RoPE/LayerNorm/RMS — реплики.
- `linear_attention`: in-proj N-split по key/value-головам; depthwise conv
  per head; рекуррентное delta-состояние — rank-local (аналог KV); out-proj —
  K-split + allreduce.
- Bias: исключается из K-split частичных GEMM; добавляется один раз в reducer.
- KV: `cldnn::kv_cache` с `num_kv_heads_r` — `[blocks, heads_r, block, hs]`;
  `VariableState` группируется (аналог CPU-агрегации состояний).

### 4.4 Runtime
- `TPCompiledModel`: k графов (`cldnn::network` на устройство+очередь) + TPPlan.
- `TPInferRequest`: входы реплицируются на все ранги, выходы с ранга 0;
  `start_async` энквьюит все ранги. Швы — device-side: publish-копия в
  staging → wait на IPC-события соседей → assemble/sum-ядро (без хост-петли
  на слой). Коллективы — OpenCL-ядра (bootstrap через `custom_gpu_primitive`),
  fp32-аккумуляция; staging — peer-device при `zeDeviceCanAccessPeer`, иначе
  USM host.

### 4.5 Вне MVP
MoE, record&replay для TP, num_streams×ранги, oneCCL, OVMS plumbing, NPU,
мультихосты.

## 5. Сессии/этапы (атомы) и гейты

| # | Этап | GATE |
|---|------|------|
| S01 | init: PLAN/STATE/NEXT + .gitignore + notes + remotes + ветка + push | docs на origin |
| S02 | L0-спайк A (Windows): enum GPU, canAccessPeer-матрица, IPC mem/event open → `docs/notes/l0-spike.md` | capability-матрица известна |
| S03 | L0-спайк B: микробенч peer vs host-USM (bw/latency) → решение G0 в STATE | дефолт коллективов зафиксирован |
| S04 | Stage 1.1 TPCoordinator: k engines, TP-ветка compile, props | TP-компиляция 2-GPU FC-мини-сети |
| S05 | Stage 1.2 TPInferRequest: реплика цельного графа на 2 GPU, выходы rank0 | e2e паритет vs 1-GPU |
| S06 | Stage 2.1 TPShardTransform-скелет: детект MatMul/FC + TPPlan | детект полон и стабилен (Qwen3.5 LM ту-тест) |
| S07 | Stage 2.2 N-split + all-gather-эквивалент (срез констант до upload) | FC/MLP-сеть паритет cos>0.999 |
| S08 | Stage 2.3 K-split + allreduce (fp32-аккум, bias в reducer), швы-события | паритет на K-split тесте |
| S09 | Stage 2.4 int4_ASYM: веса+scales/zp до group-границ (K кратно 128) | Qwen3.5-9B-int4 TP2 prefill паритет |
| S10 | Stage 3.1 per-rank KV (+GQA головы, агрегация состояний) | greedy 32 токена паритет |
| S11 | Stage 3.2 linear_attention: delta-состояния per head, conv, out-proj | паритет 32 слоя, seq ~256 |
| S12 | Stage 3.3 Hard: dynamic shapes, IPC-события, конкурентность | multi-request, batch 1..8 |
| S13 | Stage 4.1 GenAI-форк: hint pass, CB на TP2 | генерация паритет |
| S14 | Stage 4.2 Scheduler + бенч TP2 vs TP1/MULTI | TPOT ≥ 1.5× на 2×BMG |
| S15 | Stage 5.1 перф: peer-копии, ring/tree allreduce | prefill/decode отчёт |
| S16 | Stage 5.2 стабилизация: blob-cache per rank, docs, upstream-серии | матрица §6 зелень |

> **Статус (2026-10-18).** S01–S16 выполнены в коде/доках; ветка `feature/gpu-tensor-parallel`
> отправлена в `origin` (форк `savvadesogle/openvino`), `upstream` не тронут. Все стадийные гейты
> подтверждены **статически** (ревью кода/сигнатур): сборка OpenVINO и 2× GPU/GenAI недоступны в
> рабочей среде, поэтому перф-гейты S14–S16 (TPOT/TTFT) — **частичные**. Детали/блокеры/числа —
> `STATE.md`; карта артефактов и ограничений — `docs/notes/tp-artifacts.md`. Следующие этапы
> (in-graph коллективы, peer-бэкенд/события, zero-copy rank-import, upstream-серия PR) —
> `NEXT-FEATURES.md`; запуск/подтверждение гейта на железе — `NEXT-TESTING.md`.

## 6. Тест-матрица (все проверки — на Qwen3.5-9B-int4-ov)

- Паритет логитов/токенов TP1 vs TP2 (prefill, decode), greedy.
- Dynamic shapes (batch 1..8, seq 1..2048), multi-request concurrency.
- Состояния через `query_state()` roundtrip (KV + linear attention).
- Ошибки: неделимые размеры (внятное сообщение), TP на 1 GPU, PP на GPU.
- Регресс: TP-off поведение идентично upstream.
- Перф: TPOT/TTFT TP2/TP1 ≥ 1.5/1.3; памяти без дублирования весов.

## 7. Риски и меры

1. Windows L0 caps — S02/S03 gate; универсальный fallback USM host staging.
2. Фьюжны через шов — rt_info-маркировка до фьюжнов; фьюжны не пересекают швы.
3. cldnn построен вокруг 1 device — все TP*-файлы изолированы, минимальные
   точки касания в существующем коде.
4. FP16-суммы — fp32-аккум в reducer.
5. Rebase upstream — периодический merge; изоляция кода минимизирует конфликты.

## 8. Безопасность

Секреты не хранятся в репо; PAT не пишется в файлы/конфиги/историю команд;
используются credentials gh (keyring/GH_TOKEN окружения).

## 9. Структура артефактов процесса

- `PLAN.md` — этот канон (не изменяется в сессиях).
- `STATE.md` — живое состояние: прогресс, гейт-решения, блокеры, заметки.
- `NEXT.md` — промпт следующей сессии (генерируется в конце каждой сессии).
- `docs/notes/*.md` — выжимки фактов кода, спайки, бенч-отчёты.
