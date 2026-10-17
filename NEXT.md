# СЕССИЯ S02 — L0-спайк A: capability-матрица (Windows, 2+ Arc/BMG) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии)
  2) C:\openvino-tp\STATE.md — живое состояние
  3) C:\openvino-tp\docs\notes\l0-spike-plan.md — чеклист §1 (задача этой сессии)
  4) при необходимости: C:\openvino-tp\docs\notes\gpu-plugin-arch.md §5
     (что уже есть в дереве по L0/IPC) и cpu-tp-pattern.md (общий контекст TP)
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S02 (таблица в PLAN.md §5).
  Следующие этапы запрещены, даже если остаётся время/контекст. Остановка на GATE или
  блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01 done: ветка feature/gpu-tensor-parallel создана от master (tip 24e336ca);
  4 конспекта в docs/notes/; commits 7bd6c6d4 + d0d2a20c. Если ветки ещё нет в
  origin — push был заблокирован 403, см. STATE «Блокеры» (нужен gh auth login).
- Тест-модель локально: C:\openvino-tp\models\Qwen3.5-9B-int4-ov (в этой сессии не нужна).
- Железо: Windows-машина с 2+ discrete Intel GPU (Arc/Battlemage), PCIe.

## Задача S02 (полный объём этой сессии)
Read-only спайк Level Zero: зафиксировать capability-матрицу междевайсных путей
(нужна как вход для дизайна коллективов TP). Никаких измерений производительности.

1. Инструмент: новый проект tools/l0_spike/ (только новые файлы ВНЕ src/**).
   Вариант выбрать по факту: (a) standalone C++ (MSVC cl.exe, линковка ze_loader.lib,
   системные Level Zero headers level_zero/ze_api.h); (b) Python ctypes по ze_loader.dll
   (ноль сборки). Bundled zex-заголовки из src/plugins/intel_gpu/.../compute_runtime/*
   можно использовать только как справку по zex counter-based events.
2. Выполнить чеклист docs/notes/l0-spike-plan.md §1, шаги 1–6:
   - zeInit(ZE_INIT_FLAG_GPU_ONLY), zeDriverGet, версии драйвера/loader'а;
   - перечисление GPU (свойства: имя, тип, numSlices, память, numQueues);
   - zeContextCreate на каждый device;
   - zeDeviceCanAccessPeer — матрица N×N (все упорядоченные пары, вкл. i==j) +
     zeDeviceGetP2PProperties (ACCESS/ATOMICS/ATOMICS_64BIT) для пар с доступом;
   - IPC-память: zeMemAllocDevice → zeMemGetIpcHandle → zeMemOpenIpcHandle
     (sanity на своём device, затем на каждом чужом) + zeMemCloseIpcHandle;
   - IPC-events: zexCounterBasedEventCreate2 с ZEX_COUNTER_BASED_EVENT_FLAG_IPC →
     zexCounterBasedEventGetIpcHandle → zexCounterBasedEventOpenIpcHandle +
     минимальный smoke (wait на пире после signal), без таймингов.
3. Результаты — в docs/notes/l0-spike.md: шапка (дата, Windows build, GPU/драйверы,
   версии L0), таблицы из §1.7 плана, честная фиксация недоступного/непроверенного
   (коды ze_result_t; если символ не экспортируется — так и написать).
4. STATE.md: S02 = done + дата; заметки для S03 (в т.ч. сюрпризы и рабочая гипотеза
   о дефолте коллективов, решение G0 — только в S03).
5. NEXT.md — перезаписать ПОЛНЫМ промптом S03: микробенч peer-copy vs host-USM
   staging (bw/latency) + решение G0; чеклист = l0-spike-plan.md §2.
6. Commit (сообщение: "S02: L0 capability spike") → push origin feature/gpu-tensor-parallel.

## GATE S02 (критерии завершения)
- Capability-матрица + IPC-возможности (память и события) ИЗВЕСТНЫ и записаны
  в docs/notes/l0-spike.md.
- STATE.md: S02 done. NEXT.md: готовый полный промпт S03.
- Финальный ответ сессии: кратко что сделано, что показал GATE, и текст нового NEXT.md.

## Жёсткие ограничения
- НЕ МЕРИТЬ перф (тайминги/bandwidth) — это S03; S02 фиксирует только да/нет + коды.
- НЕ изменять src/** и не запускать сборки/компиляции OpenVINO.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- Если push вернёт 403 — стоп: попроси пользователя выполнить gh auth login
  (никогда не проси присылать токен текстом в чат).
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/;
  реализовывать не по плану ничего нельзя (атомарность этапа).
