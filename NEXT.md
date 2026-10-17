# СЕССИЯ S01 — init репозитория (остаток) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии)
  2) C:\openvino-tp\STATE.md — живое состояние
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S01 (таблица в PLAN.md §5).
  Следующие этапы запрещены, даже если остаётся время/контекст. Остановка на GATE или
  блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- Remotes настроены: origin = github.com/savvadesogle/openvino.git,
  upstream = github.com/openvinotoolkit/openvino.git
- master запушен: tip 85720fe2 (upstream) + 6339e518 (PLAN/STATE/NEXT) + 4c9b93c3 (STATE-аменд)
- Worktree = shallow-fetch (depth=1) полного дерева OpenVINO + наши docs;
  models/ в .gitignore
- Тест-модель локально: C:\openvino-tp\models\Qwen3.5-9B-int4-ov
  (полный снапшот; отсутствуют только openvino_mtp_model.{xml,bin} — не блокер)

## Задача S01 (полный объём этой сессии)
1. Ветка: создать feature/gpu-tensor-parallel от текущего master, переключиться на неё.
2. docs/notes/ — четыре самодостаточных конспекта (точные пути:строки брать из
   PLAN.md §3; будущие сессии НЕ должны пересобирать эти знания):
   a) cpu-tp-pattern.md — механика TENSOR_PARALLEL в intel_cpu: стримы/ранги,
      SubMemoryManager + allgather через spin-флаги, слайс FC по out-channels,
      выводы для GPU-порта;
   b) gpu-plugin-arch.md — компиляция и cldnn-граф, очереди/события/барьеры,
      propagate_constants (upload констант), точка вставки TP-пассов,
      ограничение «1 CompiledModel = 1 device»;
   c) hetero-pp.md — как HETERO делает PIPELINE_PARALLEL и почему для TP
      непригоден напрямую;
   d) l0-spike-plan.md — чеклист S02 (zeInit/enum GPU, матрица
      zeDeviceCanAccessPeer, IPC zeMemGetIpcHandle/OpenIpcHandle для памяти и
      IPC event pool) и S03 (микробенч: peer-copy vs host-USM staging,
      bw/latency).
3. STATE.md — обновить: S01 = done + дата; блокеры не трогать (они актуальны).
4. NEXT.md — перезаписать ПОЛНЫМ промптом сессии S02: цель (read-only спайк
   L0 на Windows, 2+ Arc/BMG), шаги, ограничение «не мерить перф — это S03 /
   не менять код OpenVINO», GATE «capability-матрица + IPC известны и записаны
   в docs/notes/l0-spike.md», требование обновить STATE, сгенерировать NEXT
   для S03 и запушить коммит.
5. Commit: "S01: project init notes + S02 prompt" → push -u origin
   feature/gpu-tensor-parallel.

## GATE S01 (критерии завершения)
- Ветка активна и запушена; 4 конспекта в docs/notes/ на уровне PLAN §3.
- STATE.md: S01 done. NEXT.md: готовый полный промпт S02.
- Финальный ответ сессии: кратко — что сделано, что показал GATE,
  и текст нового NEXT.md для пользователя.

## Жёсткие ограничения
- НЕ изменять src/** и не запускать сборки/компиляции OpenVINO.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- Если push вернёт 403 — стоп: попроси пользователя выполнить gh auth login
  (никогда не проси присылать токен текстом в чат).
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/;
  реализовывать не по плану ничего нельзя (атомарность этапа).
