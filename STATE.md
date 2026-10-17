# STATE.md [копия для переноса в C:\openvino-tp\STATE.md]
Канон — PLAN.md (не редактировать в сессии). Здесь: прогресс, решения, блокеры.

## Управление
| Поле | Значение |
|---|---|
| Ветка | master (S01 создаст feature/gpu-tensor-parallel) |
| Upstream tip базис | 85720fe2 (2026-10-17) |
| ОС / железо | Windows primary; 2+ Intel Arc/BMG |
| Тест-модель | models/Qwen3.5-9B-int4-ov (полная, кроме openvino_mtp_model.*) |
| Коллективы дефолт | НЕ РЕШЕНО (Gate G0 решит S03) |

## Прогресс сессий
- [x] Планирование (2026-10-17, PLAN.md готов)
- [x] Базовая инициализация репо (2026-10-17): remotes origin/upstream, shallow-fetch tip 85720fe2, commit 6339e518 (PLAN/STATE/NEXT/.gitignore) запушен в origin master
- [ ] S01 init (остаток: ветка feature/gpu-tensor-parallel, docs/notes/*, генерация NEXT.md для S02)
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

## Блокеры / открытые вопросы
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- (генерируется в конце каждой сессии)
