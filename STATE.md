# STATE.md [копия для переноса в C:\openvino-tp\STATE.md]
Канон — PLAN.md (не редактировать в сессии). Здесь: прогресс, решения, блокеры.

## Управление
| Поле | Значение |
|---|---|
| Ветка | feature/gpu-tensor-parallel (активна; S02 закоммичен локально; push в origin — см. Блокеры) |
| Upstream tip базис | 85720fe2 (2026-10-17) |
| ОС / железо | Windows primary; 2+ Intel Arc/BMG |
| Тест-модель | models/Qwen3.5-9B-int4-ov (полная, кроме openvino_mtp_model.*) |
| Коллективы дефолт | НЕ РЕШЕНО (Gate G0 решит S03) |

## Прогресс сессий
- [x] Планирование (2026-10-17, PLAN.md готов)
- [x] Базовая инициализация репо (2026-10-17): remotes origin/upstream, shallow-fetch tip 85720fe2, commit 6339e518 (PLAN/STATE/NEXT/.gitignore) запушен в origin master
- [x] S01 init (2026-10-17): ветка feature/gpu-tensor-parallel от master; docs/notes/{cpu-tp-pattern,gpu-plugin-arch,hetero-pp,l0-spike-plan}.md; STATE обновлён; NEXT — промпт S02; commit 7bd6c6d4 (push 403 — ждёт авторизацию, см. Блокеры)
- [x] S02 L0-спайк A (2026-10-17): tools/l0_spike/{l0_capability.py,README.md,l0_capability_output.txt}; результаты — docs/notes/l0-spike.md; capability-матрица зафиксирована (GATE пройден)
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
| 2026-10-17 | S02 L0-спайк A | done | capability-матрица известна и записана в docs/notes/l0-spike.md: 2×A770; canAccessPeer кросс = false; P2P кросс = 0x0; IPC-память UNSUPPORTED_FEATURE (core и ZEX_mem_ipc_handles); counter-based IPC event GetIpcHandle = OUT_OF_DEVICE_MEMORY. Решение G0 — в S03 (перф не мерялась) |

## Блокеры / открытые вопросы
- S01 push: `git push -u origin feature/gpu-tensor-parallel` → **403** "Permission to
  savvadesogle/openvino.git denied to savvadesogle". Нужно: пользователю выполнить
  `gh auth login` с доступом на запись к origin (или обновить credentials/PAT с
  правами contents:write; НЕ присылать токен текстом в чат). Локально закоммичено:
  7bd6c6d4 (S01) + этот STATE-коммит; после авторизации — повторить push.
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- S03 (следующий атом): микробенч L0 — чеклист `docs/notes/l0-spike-plan.md` §2;
  результаты/решение **G0** (дефолт коллективов) — в STATE (журнал гейтов) + кратко
  в `docs/notes/l0-spike.md`. Инструмент развивать в `tools/l0_spike/`
  (`l0_capability.py` уже резолвит zex через `zeDriverGetExtensionFunctionAddress`
  и умеет alloc/copy — переиспользовать).
- **Критично для S03** (из результатов S02): на этой конфигурации междевайсные
  P2P и IPC памяти/событий **недоступны** — см. `l0-spike.md` §6. Peer-copy
  мерять не на чем (`canAccessPeer=false`); «универсальный fallback» USM host
  staging из PLAN §4.4 становится, по сути, единственным кандидатом. Бенч должен
  подтвердить числами bw/latency device→pinned-host→device (и, если возможно,
  shared-USM) — тогда G0.
- **Осторожно**: counter-based `zexCounterBasedEvent*` — device-side примитив;
  `zeEventHostSignal` на нём даёт UNSUPPORTED_FEATURE, а host-query всегда
  SUCCESS. Для host-синхронизации швов использовать обычные `zeEvent*`
  (в S02 classic `zeEventCreate` **крашнул** драйвер из-под ctypes — при S03
  подходить аккуратно, лучше через event pool; отмечено, чтобы не потерять).
- Находки S02 (не реализовывать до своих этапов; детали — в docs/notes/l0-spike.md):
  1. `zeDeviceGetP2PProperties` в SDK 1.33.1 не имеет флага `ATOMICS_64BIT`
     (только ACCESS/ATOMICS) — учесть в дизайне allreduce.
  2. Драйвер рапортует `ZE_IPC_PROPERTY_FLAG_MEMORY|EVENT_POOL`, но фактически
     IPC не работает на Windows — не доверять `zeDriverGetIpcProperties`.
  3. `zeLoader` не экспортирует `zex*` напрямую — только через
     `zeDriverGetExtensionFunctionAddress` (важно для загрузки в плагине, S04/S12).
  4. `ZEX_mem_ipc_handles` документирован как Linux-only (official doc).
- Находки S01 (сохранены ниже, не реализовывать до своих этапов):
  1. `tools/continuous_batching` в ЭТОМ репо отсутствует — GenAI CB живёт в
     отдельном репо openvino.genai; в дереве openvino строка
     MODEL_DISTRIBUTION_POLICY только в properties.hpp (+pybind). Учесть при S13.
  2. `zeDeviceCanAccessPeer`/`zeDeviceGetP2PProperties`/`zeMemGetIpcHandle`/
     `zeMemOpenIpcHandle` НЕ объявлены в дереве intel_gpu — S02 брал системные
     L0 headers/loader. `zexCounterBasedEventGetIpcHandle/OpenIpcHandle` есть
     (zex_event.h:46-50, zex_common.h:167-204) и в плагине не используются.
  3. CPU-механика TP делает только N-split FC, K-split/allreduce отсутствует —
     allreduce(sum) для GPU проектируется с нуля (S08); CPU `split_parts` даёт
     неравные хвосты (len%n) — для int4 K-границ НЕ годится (K кратно 128).
  4. Counter-based events отключены на xe3/xe3p/unknown (ze_stream.cpp:246-250);
     на A770 (Xe-HPG) counter-based `Create2`/локальные события работают (S02),
     но IPC-экспорт их недоступен.
  5. Blob-cache GPU имеет compat-descriptor runtime requirements
     (plugin/compiled_model.cpp:97-120) — per-rank кэш различать по (rank, device)
     при S16.
