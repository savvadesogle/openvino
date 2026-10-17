# L0 capability spike — результаты S02 (Windows)

Канон: PLAN.md §4.4, §5 (S02). Чеклист: `docs/notes/l0-spike-plan.md` §1.
Метод: read-only спайк через Python `ctypes` поверх системного `ze_loader.dll`
(вариант (b) из плана; MSVC `cl.exe` в PATH отсутствует).
Инструмент: `tools/l0_spike/l0_capability.py`; сырой вывод —
`tools/l0_spike/l0_capability_output.txt`.

## 0. Шапка окружения

| Поле | Значение |
|---|---|
| Дата | 2026-10-17 |
| ОС | Windows 11 Pro for Workstations, build 10.0.22631 |
| GPU | 2× **Intel(R) Arc(TM) A770 Graphics** (DG2/ACM-G10) |
| GPU vendorId / deviceId | 0x8086 / 0x56A0 (обе карты идентичны) |
| GPU driver (Windows, WMI) | **32.0.101.9150** |
| PCI BDF | dev0 = 0000:05:00.0, dev1 = 0000:85:00.0 (разные PCIe root, gen/width из L0 не отдаются: gen=-1 x=-1) |
| L0 loader | `C:\Windows\System32\ze_loader.dll`, file version **1.33.1.0** |
| L0 `zeDriverGetApiVersion` | **1.17** |
| L0 `zeDriverGetProperties.driverVersion` | 0x01039BD6 |
| L0 headers (справка) | LevelZeroSDK 1.33.1 (`ze_api.h`), bundled `zex_*.h` из compute-runtime 4df478c |
| Python | 3.11.9 |

`zeInit(ZE_INIT_FLAG_GPU_ONLY)` → `ZE_RESULT_SUCCESS`; 1 driver, 2 GPU-устройства
(`ZE_DEVICE_TYPE_GPU`).

## 1. Перечисление GPU (шаг 2)

| # | name | type | numSlices | subslices/slice | EUs/subslice | coreClock | local mem |
|---|---|---|---|---|---|---|---|
| 0 | Intel(R) Arc(TM) A770 Graphics | 1 (GPU) | 8 | 8 | 8 | 2400 MHz | 15.56 GiB |
| 1 | Intel(R) Arc(TM) A770 Graphics | 1 (GPU) | 8 | 8 | 8 | 2400 MHz | 15.56 GiB |

Compute props (обе): maxTotalGroupSize 1024; subGroupSizes = {8,16,32}.
`maxMemAllocSize`, `maxHardwareContexts` (поля есть в `ze_device_properties_t`).

Driver IPC properties: `flags = 0x3` → `ZE_IPC_PROPERTY_FLAG_MEMORY` **и**
`ZE_IPC_PROPERTY_FLAG_EVENT_POOL` выставлены драйвером.
Интересные расширения драйвера: `ZEX_counter_based_event`,
`ZEX_mem_ipc_handles`, `ZE_experimental_event_pool_counter_based`,
`ZE_intel_experimental_context_virtual_address_space_ipc_handle`.

## 2. Контексты (шаг 3)

`zeContextCreate(driver, {stype=CONTEXT_DESC, flags=0})` на каждый device →
`ZE_RESULT_SUCCESS` для обоих.

## 3. Peer-матрица (шаг 4)

`zeDeviceCanAccessPeer(i, j)` — строки i = обращающийся, столбцы j = владелец
памяти:

| i \ j | 0 | 1 |
|---|---|---|
| **0** | **Y** | n |
| **1** | n | **Y** |

(все вызовы вернули `ZE_RESULT_SUCCESS`; «n» = `value=0`, т.е. доступа нет).
**Кросс-девайсного peer-доступа между двумя A770 нет.**

`zeDeviceGetP2PProperties(i, j)` (флаги):

| пара i→j | flags | ACCESS | ATOMICS |
|---|---|---|---|
| 0→0 | 0x3 | да | да |
| 0→1 | 0x0 | нет | нет |
| 1→0 | 0x0 | нет | нет |
| 1→1 | 0x3 | да | да |

Примечание: флага `ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS_64BIT`, упомянутого в
плане, в заголовке SDK 1.33.1 **нет** (есть только ACCESS и ATOMICS).

## 4. Сводная операционная таблица (шаги 3–6)

Обозначение «код» = `ze_result_t` (SUCCESS → 0).

| операция | i==j | i→j (0→1) | j→i (1→0) | код / вердикт |
|---|---|---|---|---|
| `zeContextCreate` | ok | ok | ok | SUCCESS |
| `zeMemAllocDevice` (1 MiB, align 64) | ok | — | — | SUCCESS |
| `zeMemGetIpcHandle` (core, device USM) | **нет** | — | — | `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` |
| `zeMemGetIpcHandle` (core, host USM) | **нет** | — | — | `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` |
| `zexMemGetIpcHandles` (ext) | **нет** | — | — | `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` |
| `zeMemOpenIpcHandle` | — | не исп. | не исп. | не проверено (нет handle от Get) |
| `zexCounterBasedEventCreate2` (без IPC) | ok | — | — | SUCCESS |
| `zexCounterBasedEventCreate2` (+IPC, 0x8/0xD/0xE) | ok | — | — | SUCCESS (создаётся) |
| `zexCounterBasedEventGetIpcHandle` | **нет** | — | — | `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (все варианты флагов) |
| `zexCounterBasedEventOpenIpcHandle` | — | не исп. | не исп. | не проверено (нет IPC handle) |

`zeMemGetIpcHandle` возвращает UNSUPPORTED_FEATURE и для **host**-USM
(`zeMemAllocHost`), т.е. это не про device/host, а про отсутствие IPC-экспорта
на Windows вообще.

## 5. Доступность символов из loader'а

- Core L0-функции (`zeDriverGet`, `zeDeviceCanAccessPeer`,
  `zeDeviceGetP2PProperties`, `zeMemGetIpcHandle` и др.) — **экспортируются**
  `ze_loader.dll` (`bind` без ошибок).
- `zex*` (`zexCounterBasedEventCreate2/GetIpcHandle/OpenIpcHandle/CloseIpcHandle`,
  `zexMemGetIpcHandles/OpenIpcHandles`) — **не экспортируются** loader'ом как
  обычные символы; получены через
  `zeDriverGetExtensionFunctionAddress` (все резолвятся). Их прототипы взяты из
  bundled `zex_event.h` и из официального дока расширения
  `MULTIPLE_IPC_HANDLES.md` (compute-runtime).

## 6. Сюрпризы и открытые вопросы

1. **Нет P2P между A770.** `canAccessPeer=false` для обеих допускающих
   ориентированных пар; P2P-флаги кросс-пар = 0. Две карты сидят на разных PCIe
   root complex (bus 05 vs 85). ⇒ Прямой peer-copy/asymmetric IPC памяти как
   основа коллективов на этой конфигурации недоступен.
2. **IPC-память на Windows не поддерживается.** И core `zeMemGetIpcHandle`
   (device и host USM), и `ZEX_mem_ipc_handles` дают `UNSUPPORTED_FEATURE`.
   Официальный док расширения прямо говорит: «Currently supported only on
   Linux. Mainly implemented for and validated on Xe HPC (PVC)». Символы при
   этом резолвятся (создаётся впечатление доступности) — но операции падают.
3. **Counter-based IPC-события не экспортируются.** `Create2` с флагом IPC
   успешен для всех вариантов (0x8 IPC-only, 0xD IMMEDIATE|HOST_VISIBLE|IPC,
   0xE NON_IMMEDIATE|HOST_VISIBLE|IPC), но `GetIpcHandle` стабильно даёт
   `OUT_OF_DEVICE_MEMORY` — межпроцессное/междевайсное разделение событий
   недоступно.
4. **Host-семантика counter-based event подозрительна.**
   `zeEventHostSynchronize(ev, 0)` возвращает SUCCESS **до** любого сигнала, а
   `zeEventHostSignal` для counter-based event даёт `UNSUPPORTED_FEATURE`. Т.е.
   host-видимость counter-based событий на этом драйвере не работает как у
   обычных событий (это **device-side** примитив). Это отдельный риск для «швов»
   TP, если опираться на host-polling.
5. **Драйвер рапортует MEMORY+EVENT_POOL IPC capability, но фактически их не
   реализует на Windows** — расхождение заявленного и фактического; полагаться
   на `zeDriverGetIpcProperties` в дизайне нельзя.
6. **PCI speed приходит неизвестной** (gen=-1, width=-1, maxBandwidth=-1) —
   информацию о линке L0 не отдаёт; для S03 bandwidth считать по факту копий.
7. Не проверено (вне спайка, отмечено честно): **кросс-процессный** IPC
   (в S02 всё в одном процессе) и работа `zeMemOpenIpcHandle` (её нельзя было
   вызвать: `zeMemGetIpcHandle` не вернул handle).

## 7. Вывод для дизайна TP (вход в S03, без решения G0)

На данной конфигурации (2×A770, Windows, L0 1.33.1, драйвер 32.0.101.9150)
**все** междевайсные пути, требующие P2P или IPC-экспорта памяти/событий,
недоступны. Значит рабочим дефолтом коллективов почти наверняка окажется
**host-USM staging** (device→pinned host→device). Но точное решение Gate **G0**
принимается только в S03 после микробенча (bandwidth/latency) — S02 перф не
меряет.

Здесь же фиксируется важное следствие для S03: **peer-copy мерять не на чем**
(`canAccessPeer=false`), поэтому бенч должен сравнивать (а) `zeMemAllocShared`/
`zeMemAllocHost` (pinned USM) staging, с явной синхронизацией событиями/барьерами
(обычными `zeEvent*`, а не counter-based IPC), и (б) любые доступные
device-to-device пути, даже если они идут через host. Гипотезу о дефолте
подтвердить числами в S03.
