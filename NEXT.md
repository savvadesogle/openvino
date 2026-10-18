# СЕССИЯ S14 — Stage 4.2 Scheduler + бенч TP2 vs TP1/MULTI (GATE TPOT ≥ 1.5×) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
Сначала полностью прочитай:
1. `C:\openvino-tp\PLAN.md` — канон (НЕ редактировать в сессии), особенно §4.4 (runtime:
   `TPCompiledModel`/`TPInferRequest`, выходы с ранга 0), §5 (строка S14: «Stage 4.2 Scheduler + бенч
   TP2 vs TP1/MULTI | TPOT ≥ 1.5× на 2×BMG»), §6 (тест-матрица: перф TPOT/TTFT TP2/TP1 ≥ 1.5/1.3,
   память без дублирования весов), §7 (численные инварианты: fp32-аккумуляция).
2. `C:\openvino-tp\STATE.md` — особенно «Заметки сессия → следующая» → блок **S14 стартует с** в записи
   S13, а также записи S13 (GenAI-путь, эмуляционные тесты, REPLICATED-ожидание), S12 (`TPTransport`
   host-USM staging, `m_infer_mutex`/`m_output_mutex`, serial-исполнение рангов) и S08/S09 (K-split +
   host reducer).
3. `C:\openvino-tp\docs\notes\genai-integration.md` — путь GenAI → TP-ветка (S13), совместимость
   compile-фрейма, тест-план паритета генерации.
4. `C:\openvino-tp\docs\notes\gpu-plugin-arch.md` §3–§4 и `C:\openvino-tp\docs\notes\l0-spike.md` §8 —
   архитектура GPU-плагина (очереди/события/`network::execute`), транспорт host-USM staging (G0).
5. `C:\openvino-tp\docs\notes\cpu-tp-pattern.md` — эталон CPU TP (ранги/стримы, агрегация состояний).

Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = **S14** (таблица PLAN.md §5). Следующие
этапы (S15 peer/ring-allreduce-перф, S16 стабилизация) ЗАПРЕЩЕНЫ. Остановка на GATE/блокере —
нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC/кросс-shared USM NA);
  **G0 = host-USM staging** (dev→pinned host→dev, ~4–5 GB/s e2e).
- S04: `TPCoordinator`/TP-ветка `compile_model`; свойства `tp_ranks`, `enable_tensor_parallel`,
  `model_distribution_policy` (+ ошибки «GPU.0 + TP», «PP на GPU», «TP на 1 GPU»).
- S05: `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы/состояние rank0);
  `create_infer_request()` → async-обёртка на `CPUStreamsExecutor`.
- S06: `TPShardTransform` классифицирует seams; RO-property `tp_seams`.
- S07: N-split + all-gather-эквивалент (`TPOutputSeam` ALL_GATHER, синтетический полный порт).
- S08: K-split + all-reduce(sum) — K-срез весов/активации, subtractive reducer (fp32).
- S09: int4_ASYM — `match_quantized_weight`, N/K-срез weight+scale+zp, sub-byte выравнивание.
- S10/S11: per-rank KV + агрегация состояний (`analyze_state_variable`, `TPStateSeam`,
  `tp_state_seams`), depthwise-conv-per-head.
- S12: динамические state-шейпы (`TPStateSeam.full_shape` = `PartialShape`), `TPTransport`
  (host-USM staging: `all_gather_axis`/`scatter_axis`/`all_reduce_sum_fp32`), конкурентность
  (`m_infer_mutex`, `m_output_mutex`). **Важно: `TPInferRequest::infer()` сейчас исполняет ранги
  ПОСЛЕДОВАТЕЛЬНО** (детерминированный порядок staging; параллелизм осознанно отложен на S14).
- S13 (2026-10-18, частичный GATE): GenAI-путь разобран (`docs/notes/genai-integration.md`), строковый
  GenAI-конфиг доходит до TP-ветки (правок не потребовалось), добавлены тесты
  `GenAiStringConfigEnablesTensorParallel`, `GenAiStyleStatefulRoundtripMatchesSingleGpu`. Реальный
  Qwen3.5-9B-int4 на TP2 по-прежнему преимущественно **REPLICATED** (coupled fused `in_proj`/
  `Concat`/`Loop`/`ScatterNDUpdate`, нет in-graph межранговых коллективов).
- Сборка OpenVINO недоступна (сабмодули пусты), GenAI-пакета и 2 GPU на сборочной машине нет →
  перф-прогон, вероятно, снова недоступен; тогда GATE частичный (дизайн scheduler + бенч-план + статическое ревью).

## Задача S14 (полный объём этой сессии)

Stage 4.2 — **Scheduler для TP-рангов + бенчмарк TP2 vs TP1/MULTI; цель GATE — TPOT ≥ 1.5× на 2×BMG.**
Точный объём:

1. **Scheduler рангов (ядро этапа).**
   - Спроектируй и реализуй планировщик исполнения k рангов TP-запроса так, чтобы per-rank sub-запросы
     шли **параллельно** (а не serial, как сейчас), сохраняя детерминизм сборки выходов/состояний.
     Точки: `TPInferRequest::infer()` (`src/plugins/intel_gpu/src/plugin/tp/tp_infer_request.cpp:176`),
     `m_infer_mutex` (сериализация одного запроса) и `m_output_mutex` (lazy-аллокация выходов).
   - Учти существующую async-инфраструктуру: `TPCompiledModel::create_infer_request()` уже даёт
     `ov::IAsyncInferRequest` на выделенном `CPUStreamsExecutor`; per-rank `CompiledModel` имеет свои
     device-очереди. Возможные подходы (выбери и обоснуй в STATE): (а) `std::async`/пул потоков на k
     рангов внутри `infer()`; (б) общий `ov::threading::IStreamsExecutor` с k задачами; (в) enqueue всех
     k рангов как async sub-запросов и общий wait. Соблюдай: **выходы/состояние — только rank-0**;
     коллективы (`TPTransport`) должны работать после завершения всех рангов; повторный `infer()`/
     конкурирующие запросы не должны гонять общий staging.
   - Не регрессируй S04–S12 (особенно конкурентность S12: независимые запросы изолированы). Форматы
     `tp_seams`/`tp_output_seams`/`tp_state_seams` стабильны.
   - Тесты: конкурентное исполнение рангов с проверкой численного паритета vs 1-GPU (расширь
     `ConcurrentRequestsMatchSingleGpu`/добавь scheduler-тест); SKIP <2 GPU.

2. **Бенчмарк TP2 vs TP1/MULTI.**
   - Подготовь бенч-скрипт/харнесс (напр. `tools/tp_bench/` или тест-утилиту), замеряющий **TTFT** и
     **TPOT** для Qwen3.5-9B-int4 через публичный API (TP2 `model_distribution_policy{TENSOR_PARALLEL}`)
     и 1-GPU (и, при наличии, штатный MULTI-девайс как референс). Фиксируй: warmup, число итераций,
     seq len, режим greedy.
   - Критерий GATE PLAN §6: **TPOT TP2/TP1 ≥ 1.5×**, TTFT ≥ 1.3×; память — без дублирования весов
     (срез до upload, PLAN §4.2). Если недостижимо из-за REPLICATED-слоёв — зафиксируй профиль, где
     время уходит (host-USM staging из G0 — узкое место, ~4–5 GB/s e2e), и оцени вклад в STATE.
   - Если бенч недоступен в среде — предоставь **бенч-план + ручной сценарий** (в `docs/notes/`) с
     описанием метрик, окружения (2× Arc A770 vs 2× BMG), команд и критериев; GATE частичный.

3. **Не регрессировать S04–S13.**
   - Паритеты S05/S07/S08/S09/S10/S11/S12/S13, негативы S04, seams S06, TP-off == upstream, GenAI-тесты S13.
   - Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S13).

4. **GATE S14.**
   - Главный GATE: **TPOT ≥ 1.5×** (TTFT ≥ 1.3×) на 2× BMG. Если железо/сборка недоступны —
     **частичный GATE**: код scheduler + бенч-обвязка + бенч-план/ручной сценарий + статическое ревью;
     честная пометка в STATE.

## GATE S14 (критерии завершения)
- Scheduler исполняет k рангов параллельно; детерминизм выходов/состояний сохранён; S04–S13 не сломаны.
- Есть бенч-обвязка (или бенч-план + ручной сценарий), измеряющая TTFT/TPOT TP2 vs TP1/MULTI.
- Достигнут TPOT ≥ 1.5× (TTFT ≥ 1.3×) ИЛИ частичный GATE с оценкой вклада REPLICATED/staging.
- STATE.md: S14 done/блокер + артефакты + решения (дизайн scheduler, узкие места перфа) + заметки → S15.
- NEXT.md — полный самодостаточный промпт S15 (перф: peer-копии, ring/tree allreduce; prefill/decode отчёт).
- Commit (сообщение: "S14: rank scheduler + TP2 vs TP1 benchmark") — **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S15+ (peer-копии, ring/tree allreduce, in-graph device-side
  коллективы как перф-оптимизация, blob-cache per-rank, upstream-серии). S14 = scheduler + бенч.
- Не коммитить `.bin`/модельные файлы; секреты не писать; `.bin` (4.6 GB) не читать целиком.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- **Форматирование**: репо пинует clang-format-18, в среде только clang-format 20.1.0 →
  whole-file прогон даёт массовый churn. **Не запускать clang-format**; стиль вручную.
- Попутные находки — в STATE «Заметки сессия → следующая» или `docs/notes/`.
- Если сборка OpenVINO недоступна (сабмодули пусты) — не изобретать обходные сборки; зафиксировать
  блокер в STATE и считать GATE частичным (код scheduler + бенч-план готовы).
