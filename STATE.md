# STATE.md [копия для переноса в C:\openvino-tp\STATE.md]
Канон — PLAN.md (не редактировать в сессии). Здесь: прогресс, решения, блокеры.

## Управление
| Поле | Значение |
|---|---|
| Ветка | feature/gpu-tensor-parallel (активна, **только локально**) |
| Remote-политика | **Работаем только локально, remote НЕ трогаем** (решение 2026-10-17): коммиты не пушим, origin/upstream не изменяем |
| Upstream tip базис | 85720fe2 (2026-10-17) |
| ОС / железо | Windows primary; 2+ Intel Arc/BMG |
| Тест-модель | models/Qwen3.5-9B-int4-ov (полная, кроме openvino_mtp_model.*) |
| Коллективы дефолт | **host-USM staging** (pinned `zeMemAllocHost` + per-device command queue; G0 решён в S03). Peer/IPC/кросс-shared USM на целевой машине недоступны; peer — runtime fast-path |

## Прогресс сессий
- [x] Планирование (2026-10-17, PLAN.md готов)
- [x] Базовая инициализация репо (2026-10-17): remotes настроены (origin/upstream); shallow-fetch tip 85720fe2; commit 6339e518 (PLAN/STATE/NEXT/.gitignore). Далее — работа только локально.
- [x] S01 init (2026-10-17): ветка feature/gpu-tensor-parallel от master; docs/notes/{cpu-tp-pattern,gpu-plugin-arch,hetero-pp,l0-spike-plan}.md; STATE обновлён; NEXT — промпт S02; commit 7bd6c6d4 (локально)
- [x] S02 L0-спайк A (2026-10-17): tools/l0_spike/{l0_capability.py,README.md,l0_capability_output.txt}; результаты — docs/notes/l0-spike.md; capability-матрица зафиксирована (GATE пройден)
- [x] S03 L0-спайк B (G0) (2026-10-17): tools/l0_spike/l0_bench.py (+ output); bw/latency-таблицы — docs/notes/l0-spike.md §8; **G0 = host-USM staging** (журнал гейтов)
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
| 2026-10-17 | S01 init | done (локально) | артефакты GATE готовы: ветка активна, 4 конспекта в docs/notes, STATE/NEXT обновлены; commit 7bd6c6d4 локально |
| 2026-10-17 | S02 L0-спайк A | done | capability-матрица известна и записана в docs/notes/l0-spike.md: 2×A770; canAccessPeer кросс = false; P2P кросс = 0x0; IPC-память UNSUPPORTED_FEATURE (core и ZEX_mem_ipc_handles); counter-based IPC event GetIpcHandle = OUT_OF_DEVICE_MEMORY. Решение G0 — в S03 (перф не мерялась) |
| 2026-10-17 | **S03 L0-спайк B → G0** | **G0 = host-USM staging** | Микробенч (docs/notes/l0-spike.md §8): единственный рабочий междевайсный путь — dev_i→pinned host→dev_j; ~7–10 GB/s/плечо, ~4–5 GB/s end-to-end (16–128 MB); локальная same-device полоса 145–192 GB/s; кросс-shared USM = OUT_OF_DEVICE_MEMORY; peer/IPC NA (S02). Дефолт коллективов TP — host-USM staging; peer — runtime fast-path по `zeDeviceCanAccessPeer` |

## Блокеры / открытые вопросы
- **Решено (2026-10-17): remote не используем.** Сессия работает исключительно в
  локальном репозитории `C:\openvino-tp`; на `origin`/`upstream` ничего не пушим и
  не изменяем. Формулировка «push» в старых записях/промптах считается снятой.
  Локальные коммиты S01/S02: 7bd6c6d4, d0d2a20c, b97946ad, e6cbc7d5 (и далее).
- Докачать `openvino_mtp_model.{xml,bin}` перед S10/S11 (не блокер).

## Заметки сессия → следующая
- S04 (следующий атом): Stage 1.1 `TPCoordinator` — k engines, TP-ветка
  `Plugin::compile_model`, properties (`ov::hint::model_distribution_policy`,
  `ov::intel_gpu::enable_tensor_parallel`, `tp_ranks`). GATE: TP-компиляция
  2-GPU мини-сети с FC. Чеклист — PLAN §4.1–4.2, строка S04 таблицы §5.
- **Вход S04 из S03 (G0)**: дефолт коллективов = **host-USM staging**
  (`zeMemAllocHost` pinned + per-device command queue и `zeCommandListAppendMemoryCopy`,
  host-sync через `zeCommandQueueSynchronize`). Междевайсный путь на этой машине
  только через host (~4–5 GB/s end-to-end на 16–128 MB; ~80 µs host-sync floor
  на малый вызов). Peer/IPC/кросс-shared USM — NA. S04 коллективы ещё **не
  реализует** (только k engines + compile + props); staging-транспорт — S08/S12.
  Детали — docs/notes/l0-spike.md §8.
- **Осторожно** (перенос из S02): counter-based `zexCounterBasedEvent*` —
  device-side примитив; `zeEventHostSignal` на нём даёт UNSUPPORTED_FEATURE, а
  host-query всегда SUCCESS. Classic `zeEventCreate` из ctypes **крашнул** драйвер
  (S02). S03 обошелся без событий вообще: `zeCommandQueueSynchronize` —
  рабочий host-sync. Для device-side швов — event pool, аккуратно (S08/S12).
- Находки S03 (не реализовывать до своих этапов; детали — docs/notes/l0-spike.md §8):
  1. Междевайсный путь только через host: staging end-to-end ~4–5 GB/s (16–128 MB);
     одна сторона d2h/h2d ~7–10 GB/s; на ≤1 MB числа кэш-шумят (не PCIe).
  2. `zeMemAllocShared` **не** кросс-девайсный на этой машине: копия между
     shared(ctx0/dev0) и dev1 → `OUT_OF_DEVICE_MEMORY`. Как транспорт не годится.
  3. `zeCommandListAppendMemoryCopy` на regular list + in-order queue +
     `zeCommandQueueSynchronize` — рабочий host-sync без событий; host-sync floor
     ~60–90 µs/вызов (аргумент за device-side синхронизацию швов, S08/S12).
  4. Host USM, аллоцированный в ctx0, доступен обеим картам (h2d host→dev1 из
     ctx0 работает) — база для единого staging-буфера ранга-агрегатора.
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
