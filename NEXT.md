# СЕССИЯ S01 — init репозитория проекта TP-GPU [копия для переноса в C:\openvino-tp\NEXT.md]

Прочти сначала: PLAN.md (канон), STATE.md. Правило: исполнить ТОЛЬКО этап S01.

## Задача S01 (единственная цель)
1. Ветка: `feature/gpu-tensor-parallel` от master.
2. `.gitignore`: `models/`, `*.log`, `*.tmp` (модельные .bin НИКОГДА не в git).
3. Файлы: STATE.md (уже записанPLAN-пакетом → обновить только при необходимости),
   NEXT.md по скелету ниже, docs/notes/: `cpu-tp-pattern.md` (конспект ссылок
   PLAN §3 по CPU), `gpu-plugin-arch.md` (конспект GPU-плагина),
   `hetero-pp.md`, `l0-spike-plan.md` (что мерить в S02/S03).
4. Целостность модели: список файлов в STATE (отсутствие openvino_mtp_model.* —
   известный нон-блокер).
5. Remotes: origin = github.com/savvadesogle/openvino.git,
   upstream = github.com/openvinotoolkit/openvino.git.
6. Commit: "Init TP-GPU project state (PLAN/STATE/NEXT + notes)" →
   push -u origin feature/gpu-tensor-parallel; в STATE отметить S01 done.

## GATE S01
Все файлы на origin; ветка активна; NEXT.md содержит полный промпт S02.

## Скелет промпта S02 (заполнить при генерации NEXT.md)
```
СЕССИЯ S02 — L0-спайк A (capability; read-only exploration)
Вход: PLAN.md, STATE.md, docs/notes/l0-spike-plan.md.
Задача: на Windows (2+ Arc/BMG) изучить Level Zero:
• zeInit/zeGetDriverInstance/zeDeviceGet: список GPU-устройств
• Матрица zeDeviceCanAccessPeer по парам устройств
• IPC: zeMemGetIpcHandle/zeMemOpenIpcHandle (память), IPC event pool
Ограничения: только изучение; результаты в docs/notes/l0-spike.md;
НЕ мерить производительность (это S03); не менять код OpenVINO.
GATE: capability-матрица известна; STATE.md: S02 done + вопросы для S03;
NEXT.md: полный промпт S03.
```

## Ограничения сессии S01
- Не изменять исходники OpenVINO; не запускать сборки.
- Git-операции в пределах C:\openvino-tp; секреты не в файлах.
