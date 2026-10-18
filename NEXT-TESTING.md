# ПРОМПТ СЕССИИ: Тестирование TP-GPU на реальном железе (подтверждение GATE)

> Самодостаточный промпт. Скопируй целиком в новую сессию на машине, где возможна сборка OpenVINO
> и есть 2× Intel GPU (+ желательно `openvino_genai`). Это НЕ новый этап разработки — цель только
> подтвердить/опровергнуть GATE §6 и зафиксировать факты. Код проекта завершён (S01–S16).

## Цель сессии
Прогнать функциональные TP-тесты и перф-бенч на реальном железе, заполнить перф-матрицу фактами и
внести финальное решение GATE в `STATE.md`. Если среда снова недоступна — честно зафиксировать блокер
и остановиться (GATE частичный), не изобретая обходных сборок.

## Сначала прочитай (в этом порядке)
1. `docs/notes/tp-artifacts.md` — карта артефактов, возможности, ограничения, **§5 «Как воспроизвести»**.
2. `docs/notes/s16-perf-matrix.md` — перф-матрица TP1/TP2/MULTI, **§4 чек-лист замеров**, **§5 критерии GATE**.
3. `docs/notes/s15-perf-report.md` — трафик/крит. путь ring, prefill/decode-профиль, §7 чек-лист.
4. `STATE.md` — журнал гейтов S01–S16, блокеры, блок **S16 done**.
5. `docs/notes/tp-bench-plan.md`, `tools/tp_bench/tp_bench.py`, `tools/tp_bench/README.md` — бенч.
6. `PLAN.md` §6 — пороги GATE (`TPOT(TP1)/TPOT(TP2) ≥ 1.5`, `TTFT(TP1)/TTFT(TP2) ≥ 1.3`; §7.4 fp32).

## Шаги

### 0. Проверка среды (STOP, если непригодна)
- `git submodule status` — если сабмодули пусты, попробовать `git submodule update --init --recursive`
  **только если сеть на этой машине разрешена**; иначе — GATE частичный, стоп.
- Тулчейн: `cl`/`cmake`/`ninja`; `python -c "import openvino as ov; print(ov.Core().get_available_devices())"`.
- Нужно ≥2 `GPU.N`. `openvino_genai` — для реальных TTFT/TPOT (иначе proxy).

### 1. Сборка
- Собрать OpenVINO (плагин GPU) и Python-пакет по `CONTRIBUTING.md`; при необходимости — функциональные
  тесты GPU. Собрать `openvino_genai`, если он нужен для бенча и доступен.

### 2. Функциональные TP-тесты (главный GATE корректности)
- Запустить: `<bin>\ov_intel_gpu_behavior_tests.exe --gtest_filter=*TensorParallel*`
  (файл `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`).
- Ожидаемые группы: S05 паритет; S06 seams; S07 N-split; S08 K-split; S09 int4; S10–S12 состояние/динамика/
  конкурентность; S13 GenAI; S14 scheduler; S15 ring-детерминизм; **S16 export/import roundtrip + guard**.
- При падении — начать с гипотез плагина/рантайма (использовать skill `ov-debug`: tensor dumps, exec graph).
  Различать: (а) ошибка сборки/инфры теста, (б) реальный баг TP, (в) флейк синхронизации.
- Зафиксировать: пройдено X/Y, список падений с коротким диагнозом.

### 3. Перф-бенч (GATE §6)
- `python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov \
   --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json`
- Гreedy (`temperature=0`). Если `openvino_genai` нет — метрики помечаются indicative proxy;
  **GATE-отношения на proxy НЕ считать**.
- Снять профиль TP2 (`get_profiling_info`, префиксы `rankN:`): доля staging vs compute.
- (Опц.) Если `zeDeviceCanAccessPeer` = true (напр. 2× BMG) — зарегистрировать
  `TPTransport::set_peer_region_copy` и повторить замер; сравнить с host-staging.

### 4. Фиксация результатов
- Заполнить `docs/notes/s16-perf-matrix.md` §2 (факты) и §5 (таблица GATE): `TTFT`, `TPOT`, отношения,
  память/ранг; указать окружение (CPU, GPU, ОС, драйвер, версия OpenVINO, warmup/iters/seqlen).
- Обновить `STATE.md`: перф-числа + решение GATE (зелёный / частичный с остатком).
- Если найден баг — завести заметку в `STATE.md` («Заметки сессия → следующая») и не лезть в фичи.

## Критерии GATE (из PLAN §6)
- `TPOT(TP1)/TPOT(TP2) ≥ 1.5` И `TTFT(TP1)/TTFT(TP2) ≥ 1.3` на 2× Intel GPU → GATE зелёный.
- Паритет TP2 vs 1-GPU по функциональным тестам обязателен независимо от перфа.
- Если пороги не достигнуты — зафиксировать потолок (REPLICATED / host-staging / нет peer) и передать
  в follow-up (см. `NEXT-FEATURES.md`), GATE частичный.

## Жёсткие ограничения
- Только локально: **remote (origin/upstream) НЕ трогать** — без push/pull/PR; `git submodule update`
  только при явном разрешении на сеть.
- `PLAN.md` не редактировать; дельты — в `STATE.md`.
- Не коммитить `.bin`/модельные файлы; `.bin` (4.6 GB) целиком не читать.
- **Не запускать clang-format** (репо пинует 18) — при правках стиль вручную.
- Атомарность: не реализовывать новые фичи в этой сессии.

## Deliverables
- Обновлённые `docs/notes/s16-perf-matrix.md` и `STATE.md` с фактическими числами и решением GATE.
- Локальный commit (напр. `"S16 test: hardware validation + gate results"`), без push.
