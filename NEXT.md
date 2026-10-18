# ПРОЕКТ ЗАВЕРШЁН (S16) · TP-GPU: Tensor Parallel на GPU в OpenVINO

**Статус: проект доведён до конца (S01–S16).** Все стадии PLAN.md §5 реализованы в коде и покрыты
функциональными тестами; финальная стабилизация (S16) добавила per-rank blob-cache, единую карту
артефактов/ограничений и план upstream-серии. **GATE по сборке/прогону не подтверждён** в этой среде
(сабмодули пусты, нет 2× GPU / OpenVINO GenAI) — это зафиксировано в `STATE.md`; прогон выполняется на
машине с рабочей сборкой.

Никакого нового этапа в этой сессии не запланировано. Ниже — что сделано и чек-лист для возобновления
на железе. Remote-политика прежняя: **только локально, без push/PR**.

## 1. Итог проекта (что сделано)

- **S01–S03**: разведка; G0 = **host-USM staging** (2× A770: peer/IPC/кросс-shared USM недоступны,
  ~4–5 GB/s e2e, host-sync floor ~60–90 µs).
- **S04–S05**: `TPCoordinator`, TP-ветка `Plugin::compile_model`, `TPCompiledModel`, `TPInferRequest`
  (k per-rank sub-запросов, реплика входов, выходы/состояние с ранга 0).
- **S06**: `TPShardTransform` — детект/классификация `MatMul`/`FullyConnected` (N_SPLIT/K_SPLIT/REPLICA),
  RO-свойство `tp_seams`.
- **S07**: N-split (срез весов) + all-gather (сборка concat на ранге 0, `TPOutputSeam`).
- **S08**: K-split (срез весов+активаций) + all-reduce(sum) fp32, реплики учитываются один раз.
- **S09**: int4_ASYM (срез weight/scale/zp, суб-байт выравнивание, fallback REPLICA).
- **S10–S12**: состояния (`TPStateSeam`, head-sharding + агрегация `query_state/set_state`), depthwise-conv,
  динамические формы; `TPTransport` (host-USM staging, коллективы); конкурентность.
- **S13**: GenAI-путь (строковый конфиг → TP-ветка; name-based IO). `docs/notes/genai-integration.md`.
- **S14**: rank scheduler (параллельное исполнение рангов) + бенч-харнесс `tools/tp_bench/tp_bench.py`.
- **S15**: peer-probe (`zeDeviceCanAccessPeer`) + ring all-reduce; `docs/notes/s15-perf-report.md`.
- **S16**: **per-rank blob-cache** (конверт `"TPSB"` = план + независимый single-device блоб на ранг;
  TP-aware `Plugin::import_model_tensor_parallel`), финальные доки
  (`docs/notes/{tp-artifacts,upstream-series,s16-perf-matrix}.md`), план upstream-серии.

Полная карта кода/артефактов и ограничений — **`docs/notes/tp-artifacts.md`** (начни с неё).
План выноса в upstream — `docs/notes/upstream-series.md`. Перф-матрица — `docs/notes/s16-perf-matrix.md`.

## 2. Известное ограничение потолка (важно для ожиданий)

Реальная модель `Qwen3.5-9B-int4` на TP2 остаётся **преимущественно REPLICATED**: KV/conv/ssm-цепочки
и coupled fused `in_proj`/`Concat`/`Loop`/`ScatterNDUpdate` не head-local, межслойные швы требуют
**in-graph** коллективов (не реализованы). Поэтому:
- **корректность**: выход/состояние ранга 0 → детерминированный паритет с 1-GPU (подтверждено тестами);
- **перф**: `TPOT(TP1)/TPOT(TP2) ≈ 1.0` (< целевых 1.5×); `TTFT` — ниже 1.3×. Достижение порогов требует
  follow-up: in-graph коллективы + шардинг реальных слоёв и/или peer-копии (на A770 недоступны).

## 3. Чек-лист возобновления (на машине со сборкой + 2× Intel GPU)

1. Подтянуть сабмодули (на целевой машине сеть разрешена): `git submodule update --init --recursive`.
2. Собрать OpenVINO и Python-пакет (+ `openvino_genai`) по `CONTRIBUTING.md`.
3. `core.get_available_devices()` → должно быть ≥2 `GPU.N`.
4. Прогнать функциональные TP-тесты:
   `--gtest_filter=*TensorParallel*` (GATE-тесты S05–S16 в
   `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`).
5. Прогнать бенч: `python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov
   --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json`.
6. Заполнить `docs/notes/s16-perf-matrix.md` §2/§5 фактами и внести `TTFT/TPOT` в `STATE.md` →
   финальное решение GATE §6.
7. (Опц.) При доступном peer-доступе (2× BMG) — зарегистрировать `TPTransport::set_peer_region_copy`
   и повторить замер.
8. (Follow-up, отдельные этапы) in-graph межранговые коллективы; шардинг реальных Qwen-слоёв;
   device-side события; zero-copy rank-import. См. `docs/notes/upstream-series.md` §«Что НЕ входит».

## 4. Как читать оставшиеся материалы

1. `docs/notes/tp-artifacts.md` — карта артефактов, возможности, ограничения, воспроизведение.
2. `docs/notes/upstream-series.md` — серия PR для upstream (слайсы/заголовки/зависимости/тесты).
3. `docs/notes/s16-perf-matrix.md` — перф-матрица (аналитика + чек-лист).
4. `docs/notes/s15-perf-report.md` — трафик/крит. путь ring, prefill/decode-профиль.
5. `STATE.md` — журнал гейтов S01–S16, блокеры, заметки сессий. `PLAN.md` — канон (не редактировать).

## 5. Жёсткие ограничения (сохраняются)

- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/PR, без `git submodule update`
  в средах без явного разрешения на сеть.
- `PLAN.md` не редактировать; дельты — в `STATE.md`.
- Не коммитить `.bin`/модельные файлы; `.bin` (4.6 GB) не читать целиком.
- **Не запускать clang-format** (репо пинует 18, в среде 20.1.0 → churn); стиль вручную.
- Fallback при неделимости/небезопасном потребителе → репликация БЕЗ throw (как S07–S16).
