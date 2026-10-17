# STATE.md [копия для переноса в C:\openvino-tp\STATE.md]
Канон — PLAN.md (не редактировать в сессии). Здесь: прогресс, решения, блокеры.

## Управление
| Поле | Значение |
|---|---|
| Ветка | feature/gpu-tensor-parallel (активна с S01; push 403 — ждёт gh auth login) |
| Upstream tip базис | 85720fe2 (2026-10-17) |
| ОС / железо | Windows primary; 2+ Intel Arc/BMG |
| Тест-модель | models/Qwen3.5-9B-int4-ov (полная, кроме openvino_mtp_model.*) |
| Коллективы дефолт | НЕ РЕШЕНО (Gate G0 решит S03) |

## Прогресс сессий
- [x] Планирование (2026-10-17, PLAN.md готов)
- [x] Базовая инициализация репо (2026-10-17): remotes origin/upstream, shallow-fetch tip 85720fe2, commit 6339e518 (PLAN/STATE/NEXT/.gitignore) запушен в origin master
- [x] S01 init (2026-10-17): ветка feature/gpu-tensor-parallel от master; docs/notes/{cpu-tp-pattern,gpu-plugin-arch,hetero-pp,l0-spike-plan}.md; STATE обновлён; NEXT — промпт S02; commit 7bd6c6d4 (push 403 — ждёт авторизацию, см. Блокеры)
- [ ] S02 L0-спайк A
- [ ] S03 L0-спайк B (G0)
- [ ] S04 Stage1.1 TPCoordinator
- [ ] S05 Stage1.2 TPInferRequest
- [ ] S06 Stage2.1 TPShardTransform-скелет
- [ ] S07 Stage2.2 N-split + all-gather
- [ ] S08 Stage2.3 K-split + allreduce
- [ ] S09 Stage2.4 int4_ASYM
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
| 2026-10-17 | S01 init | done, кроме push (403) | артефакты GATE готовы: ветка активна, 4 конспекта в docs/notes, STATE/NEXT обновлены; критерий «запушена» отложен до `gh auth login` |

## Блокеры / открытые вопросы
- S01 push: `git push -u origin feature/gpu-tensor-parallel` → **403** "Permission to
  savvadesogle/openvino.git denied to savvadesogle". Нужно: пользователю выполнить
  `gh auth login` с доступом на запись к origin (или обновить credentials/PAT с
  правами contents:write; НЕ присылать токен текстом в чат). Локально закоммичено:
  7bd6c6d4 (S01) + этот STATE-коммит; после авторизации — повторить push.
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- S02: следующий атом — L0-capability спайк; чеклист = docs/notes/l0-spike-plan.md §1;
  результат писать в docs/notes/l0-spike.md. Нужна машина с 2+ Arc/BMG.
- Находки S01 (не реализовывать до своих этапов; детали — в docs/notes/):
  1. `tools/continuous_batching` в ЭТОМ репо отсутствует — GenAI CB живёт в
     отдельном репо openvino.genai; в дереве openvino строка
     MODEL_DISTRIBUTION_POLICY только в properties.hpp (+pybind). Учесть при S13.
  2. `zeDeviceCanAccessPeer`/`zeDeviceGetP2PProperties`/`zeMemGetIpcHandle`/
     `zeMemOpenIpcHandle` НЕ объявлены в дереве intel_gpu — S02 берёт системные
     L0 headers/loader. `zexCounterBasedEventGetIpcHandle/OpenIpcHandle` есть
     (zex_event.h:46-50, zex_common.h:167-204) и в плагине не используются.
  3. CPU-механика TP делает только N-split FC, K-split/allreduce отсутствует —
     allreduce(sum) для GPU проектируется с нуля (S08); CPU `split_parts` даёт
     неравные хвосты (len%n) — для int4 K-границ НЕ годится (K кратно 128).
  4. Counter-based events отключены на xe3/xe3p/unknown (ze_stream.cpp:246-250);
     на BMG (Xe2) должны работать — S02 проверит фактически (capability-матрица).
  5. Blob-cache GPU имеет compat-descriptor runtime requirements
     (plugin/compiled_model.cpp:97-120) — per-rank кэш различать по (rank, device)
     при S16.
