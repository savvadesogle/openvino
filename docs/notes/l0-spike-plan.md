# L0-спайк: план S02 (capability) и S03 (микробенч)

Создан в S01 как чеклист. S02 заполнит **результаты** в `docs/notes/l0-spike.md`
(отдельный файл; этот файл остаётся планом). Канон — PLAN.md §4.4, §5 (S02/S03).

Цель спайков: зафиксировать (S02) и измерить (S03) возможности Level Zero на
Windows для междевайсных коллективов TP на 2+ discrete Intel GPU (Arc/BMG).

## 0. Общие требования к спайку

- **Read-only относительно OpenVINO**: `src/**` не менять, сборку OpenVINO не
  запускать. Спайк — самостоятельный проект вне дерева плагина, предлагается
  `tools/l0_spike/` в этом репо (новые файлы вне `src/**` — можно).
- Вариант исполнения (выбрать в S02 по факту): (a) standalone C++ (cl.exe/MSVC,
  линковка `ze_loader.lib`), заголовки — системные `level_zero/ze_api.h`;
  (b) ctypes-обёртка `ze_loader.dll` из Python (быстрее всего, ноль сборки).
  Bundled-заголовки `src/plugins/intel_gpu/src/runtime/ze/compute_runtime/*`
  использовать только как справку по `zex` counter-based events; они НЕ содержат
  peer/IPC memory API.
- Не мерить производительность в S02 (это S03). S02 фиксирует только
  «да/нет» и коды `ze_result_t`.
- Секреты/токены в файлы не писать.

## 1. Чеклист S02 — capability-матрица

Порядок шагов (все результаты записать в `docs/notes/l0-spike.md`):

1. **Инициализация**: `zeInit(ZE_INIT_FLAG_GPU_ONLY)`; `zeDriverGet`;
   `zeDriverGetApiVersion` / версия драйвера и loader'а (`ze_loader.dll`).
   Записать версии ОС/драйвера в шапку результата.
2. **Перечисление**: для каждого driver — `zeDeviceGet`; для каждого device:
   `zeDeviceGetProperties` (name, type, `deviceId`, `numSlices`),
   `zeDeviceGetMemoryProperties` (local mem size), `zeDeviceGetComputeProperties`
   (`numQueues`). Ожидание: 2+ GPU с `ZE_DEVICE_TYPE_GPU`.
   (Для справки: как это делает плагин — `gpu-plugin-arch.md` §5,
   `ze_device_detector.cpp:72-100`.)
3. **Контексты**: `zeContextCreate` на каждый device (без флагов).
   Записать: успех/ошибка по каждому device.
4. **Peer-матрица** (для всех упорядоченных пар i→j, включая i==j):
   - `zeDeviceCanAccessPeer(dev_i, dev_j, &canAccess)` — точную сигнатуру
     взять из системного заголовка; записать булев результат в матрицу N×N;
   - для пар с `canAccess == true`: `zeDeviceGetP2PProperties(dev_i, dev_j, &props)`
     — флаги `ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS`,
     `..._ATOMICS`, `..._ATOMICS_64BIT`: отдельная матрица/таблица.
   - Если вызовы недоступны (не экспортируются loader'ом) — зафиксировать
     это как факт с кодом ошибки `dlsym/GetProcAddress`.
5. **IPC-память** (device USM):
   - владелец: `zeMemAllocDevice(ctx_i, &desc, size=1MB, align=64, dev_i, &ptr_i)`;
   - `zeMemGetIpcHandle(ctx_i, ptr_i, &handle)` → код `ze_result_t`;
   - открытие: (a) на своём же device i (sanity):
     `zeMemOpenIpcHandle(ctx_i, dev_i, handle, 0, &ptr_i2)`; (b) на каждом
     другом device j: `zeMemOpenIpcHandle(ctx_j, dev_j, handle, 0, &ptr_j)`;
     записать коды.
     Ожидаемая семантика: открытие возможно только с device, имеющим peer
     access (иначе `ZE_RESULT_ERROR_UNSUPPORTED`/`INVALID_ARGUMENT`); проверить
     фактом.
   - закрытие: `zeMemCloseIpcHandle` (проверить доступность символа).
   - Проверять **одно-контекстную** (all devices могут быть в разных ctx) и
     **одно-процессную** семантику на Windows; кросс-процессный обмен — вне
     спайка (отметить как «не проверено»).
6. **IPC-события** (counter-based, `zex`):
   - заголовки/прототипы: `zex_common.h:167-204` (флаг
     `ZEX_COUNTER_BASED_EVENT_FLAG_IPC = ZE_BIT(3)`) и `zex_event.h:40-52`
     (`zexCounterBasedEventCreate2`, `zexCounterBasedEventGetIpcHandle`,
     `zexCounterBasedEventOpenIpcHandle`, `zexCounterBasedEventCloseIpcHandle`);
   - создать событие с флагом IPC: `zexCounterBasedEventCreate2(ctx_i, dev_i,
     &desc, &ev_i)`, где `desc.flags |= ZEX_COUNTER_BASED_EVENT_FLAG_IPC`
     (+ при необходимости EXTERNAL_SYNC_ALLOC / EXTERNAL_STORAGE для
     attached-счётчика — см. `zex_common.h:216-236`);
   - `zexCounterBasedEventGetIpcHandle(ev_i, &ipc)` → ok?;
   - `zexCounterBasedEventOpenIpcHandle(ctx_j, ipc, &ev_j)` → ok? (для каждого
     peer j); sanity: wait на ev_j после signal на ev_i (host polling
     `zeEventHostSynchronize`/device wait — минимальный smoke, без таймингов).
   - Записать доступность символов из loader (экспортируются ли zex-функции).
     В плагине они объявлены, но не используются (gpu-plugin-arch.md §5).
7. **Запись результата**: `docs/notes/l0-spike.md` — таблица:
   - шапка: дата, Windows build, GPU модели/драйверы, L0 loader/драйвер версии,
     имена устройств из `zeDeviceGetProperties`;
   - таблица «пара i→j : canAccessPeer | P2P flags»;
   - таблица «операция : i==j : i→j : j→i : код результата» для
     ContextCreate/AllocDevice/IPC mem open/IPC event open;
   - выводы-сюрпризы и открытые вопросы (например, что недоступно из loader'а).

## 2. Чеклист S03 — микробенч (план; исполнять только в S03)

1. Оценка (a) **peer-copy**: `zeCommandListAppendMemoryCopy` между
   `ptr_j` (открытой IPC-памятью) и локальным USM; (b) **host-USM staging**:
   copy device→host (pinned `zeMemAllocHost`) → host→device на другом GPU.
   Размеры: 64KB, 1MB, 16MB, 128MB; обе стороны (i→j и j→i).
2. Метрики: эффективный bandwidth (GB/s) и latency (µs) на малых размерах
   (4KB/64KB); N итераций, warmup, медиана/перцентили; фиксировать, какой
   командный path использован (immediate list / copy engine).
3. Latency IPC-событий: signal (kernel на i) → наблюдаемое завершение wait на j;
   измерить для нескольких сценариев (host-visible polling vs device-side wait,
   если возможно).
4. Deliverable: таблица результатов + решение **Gate G0** (дефолт коллективов:
   peer vs host-USM staging) — записать в STATE.md (журнал гейтов) и кратко в
   `l0-spike.md`.
5. Ограничения: S03 тоже не меняет `src/**` и код OpenVINO; спайк остаётся в
   `tools/l0_spike/`.

## 3. Что S02 обязан оставить после себя

- `docs/notes/l0-spike.md` с capability-матрицей (§1.7) — вход для S03.
- Обновлённый STATE.md (S02 = done, заметки).
- NEXT.md — полный промпт S03 (микробенч + G0).
- Коммит в `feature/gpu-tensor-parallel` + push.
