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

---

# L0 microbench — результаты S03 (Windows)

Канон: PLAN.md §4.4, §5 (S03). Чеклист: `docs/notes/l0-spike-plan.md` §2.
Метод: тот же read-only спайк через Python `ctypes` поверх `ze_loader.dll`.
Инструмент: `tools/l0_spike/l0_bench.py` (переиспользует типы/loader-helpers из
`l0_capability.py`); сырой вывод — `tools/l0_spike/l0_bench_output.txt`.
Окружение идентично S02 (§0 выше): 2× Arc A770, Win 11 22631, драйвер
32.0.101.9150, L0 loader 1.33.1, apiVersion 1.17. Дата прогона: 2026-10-17.

## 8.1 Метод и командный path

- Копии: `zeCommandListAppendMemoryCopy` на **обычном (не immediate) command
  list** + in-order `zeCommandQueue`, синхронизация `zeCommandQueueSynchronize
  (UINT64_MAX)` (host-blocking). События `zeEvent*` **сознательно не
  использованы**: в S02 classic `zeEventCreate` из ctypes крашил драйвер
  (STATE.md). Path зафиксирован как «regular command list + in-order queue +
  queue synchronize».
- Размеры: 4 KB, 64 KB, 1 MB, 16 MB, 128 MB; обе стороны.
- На размер: bandwidth ≈ median per-op при batch (128/128/32/2/1 копий на один
  sync; batch амортизирует overhead сабмита), latency — median одиночной копии
  (host-observed, включает сабмит+sync). Warmup 3, reps 20 (≤1 MB) / 8 (>1 MB).
- Аллокации: на каждый GPU по 2 device-USM буфера (src/dst, 128 MB каждый);
  pinned host USM (`zeMemAllocHost`, ctx0) 128 MB; shared USM
  (`zeMemAllocShared`, ctx0/dev0) 128 MB.
- Корректность: h2d→d2h roundtrip 4 KB на каждом GPU → **MATCH** (данные
  реально ходят), см. вывод.

## 8.2 Какие пути есть, а какие NA

| путь | статус на этой машине | примечание |
|---|---|---|
| peer i→j напрямую (device USM) | **NA** | `zeDeviceCanAccessPeer=false` (S02 §3); cross P2P flags = 0x0 |
| IPC memory / IPC events | **NA** | S02 §4–6: `UNSUPPORTED_FEATURE` / `OUT_OF_DEVICE_MEMORY` |
| shared USM кросс-девайс (dev1↔shared dev0) | **NA (код ошибки)** | `zeCommandListAppendMemoryCopy` → `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` |
| **host-USM staging** (dev_i → pinned host → dev_j) | **есть** | единственный рабочий междевайсный путь; роут помечен явно |
| same-device dev→dev | есть | baseline локальной полосы |
| shared USM same-device | есть | только в пределах устройства-владельца |

## 8.3 Эффективный bandwidth, GB/s (median per-op)

| путь (route) | 4 KB | 64 KB | 1 MB | 16 MB | 128 MB |
|---|---|---|---|---|---|
| same-dev GPU0→GPU0 (device USM) | 2.86 | 70.70 | 321.25 | 145.57 | 190.95 |
| same-dev GPU1→GPU1 (device USM) | 2.92 | 77.60 | 336.55 | 152.49 | 191.74 |
| dev0→pinned host (d2h) | 2.86 | 66.13 | 148.50 | 10.21 | 10.47 |
| dev1→pinned host (d2h) | 2.99 | 71.97 | 155.81 | 7.40 | 7.28 |
| pinned host→dev0 (h2d) | 2.85 | 67.22 | 143.86 | 9.56 | 9.84 |
| pinned host→dev1 (h2d) | 2.92 | 71.70 | 148.90 | 9.59 | 9.32 |
| **staging 0→host→1** (кросс) | 0.25 | 6.16 | 44.69 | 4.78 | 4.90 |
| **staging 1→host→0** (кросс) | 0.37 | 6.18 | 44.88 | 4.03 | 4.16 |
| shared-wr dev0→shared | 4.51 | 72.41 | 292.92 | 146.24 | 191.53 |
| shared-rd shared→dev0 | 4.56 | 71.42 | 319.11 | 147.33 | 191.07 |
| shared-wr dev1→shared (кросс) | NA | NA | NA | NA | NA (OUT_OF_DEVICE_MEMORY) |
| shared-rd shared→dev1 (кросс) | NA | NA | NA | NA | NA (OUT_OF_DEVICE_MEMORY) |

Наблюдения:
- **≤1 MB узлы «кэш-шумят»**: те же буферы переиспользуются, 1 MB укладывается
  в кэш → 150–330 GB/s не отражают PCIe. **Реалистична полоса на 16/128 MB.**
- **Реальный междевайсный канал = PCIe через host**: одна сторона d2h/h2d
  ~7–10 GB/s (16–128 MB), staging end-to-end ~4–5 GB/s (сумма двух
  последовательных плеч).
- Локальная (same-device) полоса 145–192 GB/s на 16/128 MB — верхняя граница
  того, что мог бы дать peer, будь он доступен (≈20–40× быстрее staging).
- **shared USM не работает кросс-девайс** (dev1 не видит аллокацию ctx0/dev0:
  `OUT_OF_DEVICE_MEMORY`) — как и ожидалось при отсутствии peer-доступа. Значит
  `zeMemAllocShared` не является междевайсным путём на этой конфигурации.
- h2d на dev1 из host-USM, аллоцированного в ctx0, **работает** (host USM
  доступен обеим картам) — staging корректен.

## 8.4 Latency одиночной копии (host-observed, median µs)

| путь | 4 KB | 64 KB | 1 MB |
|---|---|---|---|
| same-dev GPU0→GPU0 | 89.1 | 80.9 | 84.9 |
| same-dev GPU1→GPU1 | 81.5 | 59.6 | 79.0 |
| dev0→host | 89.2 | 87.2 | 181.6 |
| host→dev0 | 89.7 | 88.3 | 190.8 |
| host→dev1 | 80.2 | 66.3 | 174.8 |

- На 4 KB / 64 KB latency ≈ **60–90 µs** для всех путей — это **floor
  host-сабмита+sync**, а не время передачи (данные физически копируются
  ~1.4 µs). Т.е. host-синхронный коллективный шов стоит десятки µs на вызов.
- Staging latency одиночки не мерилась (2 стадии + 2 sync), но по throughput
  видно, что на 128 MB staging = 27–32 ms на сообщение.
- Для «швов» TP это аргумент за **device-side синхронизацию** (события/барьеры
  между enqueue и wait, без host-polling на слой) — вынести в S12/S08. Здесь
  counter-based events device-side (S02), IPC-события — NA.

## 8.5 Вывод для G0 (решение — в STATE.md, журнал гейтов)

Единственный рабочий междевайсный путь на целевой машине — **host-USM staging**
(`zeMemAllocHost` pinned + per-device command queue, локальная копия через
`zeCommandListAppendMemoryCopy`). Полоса: ~7–10 GB/s на плечо, **~4–5 GB/s
end-to-end** на 16–128 MB; на малых размерах доминирует host-сабмит/sync
(~80 µs/вызов). Peer-path и IPC недоступны (S02), shared-USM кросс-девайс тоже
недоступен (S03). ⇒ **G0: дефолт коллективов TP = host-USM staging**;
peer-путь проектируется как runtime-detect fast-path (`zeDeviceCanAccessPeer`),
но на этой конфигурации он неактивен.

## 9. Применение G0 в TP-транспорте (S12)

S12 вводит `TPTransport` (`src/plugins/intel_gpu/src/plugin/tp/tp_transport.cpp`) — единственную точку
межранговых коллективов TP, реализующую G0:

- **Вселенная тензоров**: коллективы принимают только host-visible `ov::ITensor`; device/remote-тензоры
  отклоняются (`OPENVINO_ASSERT`). Per-rank sub-запросы GPU-плагина отдают host-тензоры (I/O по умолчанию
  host-visible), так что граница совпадает с G0-контрактом.
- **Kinds**: `TPTransportKind::HOST_STAGING` (дефолт, работает везде) и `PEER` (runtime fast-path hook,
  на 2× A770 неактивен — `tp_transport_capabilities()` кэширует `peer_available=false`, `ipc_available=false`).
- **Коллективы**: `all_gather_axis` (конкатенация per-rank регионов по оси), `scatter_axis` (обратное —
  раздача полного состояния на шарды), `all_reduce_sum_fp32` (fp32-сумма частичных + «term один раз» для
  replicated bias/residual, PLAN §4.3/§7.4). Owned `m_staging` переиспользуется (без аллокации на инференс).
- **Почему не device-side события**: S02/S03 показали, что counter-based event + `zeEventHostSignal` =
  UNSUPPORTED на этой платформе, а classic `zeEventCreate` из ctypes крашил драйвер; `zeEventPool` —
  единственный безопасный путь, но требует уровня драйвера GPU-плагина. S12 сознательно ограничен корректностью
  (host-sync на границе), device-side fast-path — S15 (перф), true in-graph inter-node коллективы — вне S12.
- **Атомарность/перф**: коллективы на хосте дают ~4–5 GB/s e2e (см. §8.3); для реального Qwen это означает, что
  полный per-rank шардинг при N>1 экономически оправдан только при перф-профиле S15, а не в S12 (приоритет —
  корректность/паритет).
