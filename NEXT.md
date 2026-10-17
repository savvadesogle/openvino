# СЕССИЯ S03 — L0-спайк B: микробенч путей междевайсной передачи + решение G0 · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии)
  2) C:\openvino-tp\STATE.md — живое состояние (особенно «Заметки сессия → следующая»)
  3) C:\openvino-tp\docs\notes\l0-spike-plan.md — чеклист §2 (задача этой сессии)
  4) C:\openvino-tp\docs\notes\l0-spike.md — результаты S02 (capability-матрица; вход в S03)
  5) при необходимости: gpu-plugin-arch.md §5 (что уже есть в дереве по L0/IPC)
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S03 (таблица в PLAN.md §5).
  Следующие этапы запрещены, даже если остаётся время/контекст. Остановка на GATE или
  блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01 done: ветка feature/gpu-tensor-parallel; 4 конспекта в docs/notes/.
- S02 done (2026-10-17): `tools/l0_spike/l0_capability.py` (+ README, + captured output);
  результаты в `docs/notes/l0-spike.md`. ГЛАВНЫЕ ФАКТЫ S02 (не переоткрывать):
  * Железо: 2× Intel Arc A770 (DG2, deviceId 0x56A0), Windows 11 build 22631,
    драйвер 32.0.101.9150, L0 loader 1.33.1, apiVersion 1.17.
  * `zeDeviceCanAccessPeer` кросс-девайс = **false**; P2P flags кросс = 0x0.
    ⇒ peer-copy как таковой **не на чем мерять**.
  * IPC-память на Windows **недоступна**: core `zeMemGetIpcHandle` и
    `ZEX_mem_ipc_handles` → `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`.
  * Counter-based IPC-события: `Create2` ok, но `GetIpcHandle` →
    `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. Символы `zex*` — только через
    `zeDriverGetExtensionFunctionAddress`.
  * Counter-based event — device-side: `zeEventHostSignal` → UNSUPPORTED_FEATURE,
    host-query всегда SUCCESS. Обычный `zeEventCreate` из ctypes **крашил** драйвер.
- Тест-модель в этой сессии НЕ нужна.

## Задача S03 (полный объём этой сессии)
Микробенч междевайсной передачи данных на 2×A770 (Windows L0) и **решение Gate G0**
(дефолтный путь коллективов TP). Всё read-only относительно OpenVINO; инструмент — в
`tools/l0_spike/` (новые файлы вне `src/**`).

1. Расширить `tools/l0_spike/` (переиспользовать `l0_capability.py`: res/alloc/copy
   helpers; новый, напр. `l0_bench.py`). Никаких изменений `src/**`, сборки OpenVINO.
2. Поскольку S02 показал отсутствие peer и IPC на этой машине, бенч строить вокруг
   реально доступных путей и **явно** это фиксировать (это ожидаемый, а не ошибочный ход):
   - (A) **host-USM staging**: device i → pinned host USM (`zeMemAllocHost`) → device j.
     Отдельно мерить device→host и host→device; считать end-to-end.
     Использовать `zeCommandListAppendMemoryCopy` на immediate-списках per-device
     (создать command list/queue на каждом GPU) **и/или** `zeMemAllocShared`;
   - (B) **shared USM** (если поддерживается): аллокация `zeMemAllocShared` и
     доступ с оба device — проверить/померить (может тоже упираться в отсутствие
     peer access — тогда зафиксировать);
   - (C) **same-device copy** (device→device в пределах одного GPU) как baseline
     локальной полосы;
   - (D) если удастся получить любой кросс-девайсный путь (даже через host) —
     обязательно пометить, через что он идёт.
   - Peer-copy (i→j напрямую) — записать как **NA** со ссылкой на S02
     (`canAccessPeer=false`); не выдумывать.
3. Размеры: 4 KB, 64 KB, 1 MB, 16 MB, 128 MB; обе стороны (0→1 и 1→0).
   Warmup + N итераций; фиксировать медиану/перцентили; эффективный bandwidth
   (GB/s) и latency (µs) на малых размерах. Указать использованный path
   (immediate list / copy engine / host memcpy).
4. Latency синхронизации (по возможности): событие/барьер между enqueue и wait;
   host-polling vs device-side. Counter-based events — только device-side (S02);
   для host-синхронизации осторожно с `zeEvent*` (краш в S02 — см. STATE).
5. Deliverable:
   - таблицы результатов в `docs/notes/l0-spike.md` (дописать раздел S03; шапку с
     avg-числами и path'ами);
   - **решение Gate G0** — дефолт коллективов (host-USM staging vs иное) — в STATE.md
     (журнал гейтов, строка «S03 … G0») + краткая сводка в `l0-spike.md`;
   - обновить поле STATE «Коллективы дефолт».
6. NEXT.md — перезаписать ПОЛНЫМ промптом S04 (Stage 1.1 TPCoordinator: k engines,
   TP-ветка compile, props; чеклист — PLAN §4.1–4.2, S04 в таблице §5).
7. Commit (сообщение: "S03: L0 microbench + G0") → push origin feature/gpu-tensor-parallel.

## GATE S03 (критерии завершения)
- bw/latency-таблицы для доступных путей записаны в `docs/notes/l0-spike.md`.
- Решение **G0** (дефолт коллективов) зафиксировано в STATE.md (журнал гейтов) и
  отражено в поле «Коллективы дефолт».
- STATE.md: S03 done. NEXT.md: готовый полный промпт S04.
- Финальный ответ сессии: кратко что сделано, что показал GATE, и текст нового NEXT.md.

## Жёсткие ограничения
- НЕ изменять `src/**`; не запускать сборки/компиляции OpenVINO.
- Спайк остаётся в `tools/l0_spike/`; не коммитить .bin/модельные файлы; секреты не
  писать в файлы/логи/историю.
- Если push вернёт 403 — стоп: попроси пользователя выполнить `gh auth login`
  (никогда не проси присылать токен текстом в чат).
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/;
  реализовывать не по плану ничего нельзя (атомарность этапа).
