# СЕССИЯ S04 — Stage 1.1 TPCoordinator: k engines, TP-ветка compile, props · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §3 (ключевые
     факты кода), §4.1–4.2 (дизайн API/compile), §5 (строка S04), §7 (риски).
  2) C:\openvino-tp\STATE.md — живое состояние (особенно «Заметки сессия → следующая»;
     поле «Коллективы дефолт» и «Находки S03»).
  3) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — конспект архитектуры GPU-плагина
     (точки касания: `plugin/plugin.cpp`, `remote_context.cpp`, `compiled_model.cpp`,
     `transformations_pipeline.cpp`).
  4) C:\openvino-tp\docs\notes\l0-spike.md — результаты L0-спайков S02/S03 (для контекста
     коллективов; в S04 коллективы ещё НЕ реализуются).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S04 (таблица PLAN.md §5).
  Следующие этапы (S05+) запрещены, даже если остаётся время/контекст. Остановка на GATE
  или блокере — нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01 done: ветка feature/gpu-tensor-parallel; 4 конспекта в `docs/notes/`
  (cpu-tp-pattern, gpu-plugin-arch, hetero-pp, l0-spike-plan).
- S02 done (2026-10-17): capability-спайк L0 (`tools/l0_spike/l0_capability.py`),
  результаты — `docs/notes/l0-spike.md` §1–7. Главное: 2× Arc A770 (DG2, 0x56A0),
  Windows 11 22631, драйвер 32.0.101.9150, L0 loader 1.33.1, apiVersion 1.17;
  `zeDeviceCanAccessPeer` кросс = false; IPC памяти/событий нет.
- S03 done (2026-10-17): микробенч (`tools/l0_spike/l0_bench.py`), результаты —
  `docs/notes/l0-spike.md` §8. **Gate G0 РЕШЁН: дефолт коллективов = host-USM
  staging** (единственный рабочий междевайсный путь: dev→pinned host→dev, ~7–10 GB/s
  на плечо, ~4–5 GB/s end-to-end на 16–128 MB; peer/IPC/кросс-shared USM — NA).
  В S04 транспорт коллективов ещё НЕ строится.
- Тест-модель: `models/Qwen3.5-9B-int4-ov/` — в S04 НЕ нужна (GATE на мини-сети с FC).

## Задача S04 (полный объём этой сессии)
Stage 1.1 — **TPCoordinator**: перечисление GPU, k per-rank engines/контекстов, TP-ветка
в `Plugin::compile_model`, свойства. Цель GATE — **TP-компиляция 2-GPU мини-сети** (модель
с одним/несколькими FullyConnected), т.е. `core.compile_model(mini_model, "GPU",
{model_distribution_policy: TENSOR_PARALLEL})` отрабатывает и создаёт k рангов;
корректность исполнения (паритет) — ещё НЕ требуется (S05).

Работать в `src/plugins/intel_gpu/**` (это уже НЕ read-only: S04 — первый этап с кодом).
Не запускать сборку, если она недоступна/слишком тяжёлая — тогда зафиксировать в STATE,
что собрать не удалось, и оставить код+тест-план; GATE считать частично пройденным с явной
пометкой. Если есть среда сборки — минимально проверить компиляцию целевых файлов.

1. **Свойства/API** (PLAN §4.1):
   - Точки: `Plugin::compile_model` (`plugin/plugin.cpp:267-306`), `PluginConfig`/
     `ExecutionConfig`, перечисление устройств `m_device_map` (`plugin/plugin.cpp:227-258`).
   - Публичный API уже есть: `src/inference/include/openvino/runtime/properties.hpp:436-488`
     (`ModelDistributionPolicy::{TENSOR_PARALLEL|PIPELINE_PARALLEL}`). В intel_gpu сейчас
     нет ни одного упоминания — добавить чтение `ov::hint::model_distribution_policy`.
   - Ввести внутренний `ov::intel_gpu::enable_tensor_parallel` и read-only
     `ov::intel_gpu::tp_ranks` (по образцу CPU — см. `docs/notes/cpu-tp-pattern.md`).
     `get_property` должен отдавать TP-отчёт. Понятные ошибки для несовместимых
     комбинаций («GPU.0» + TP, TP на 1 GPU, PP на GPU).
2. **TPCoordinator** (PLAN §4.2, §4.4):
   - Перечислить GPU (`m_device_map`), выбрать k (MVP: TP=2), создать k per-rank
     `RemoteContextImpl`/engine (это НЕ один `CompiledModel` на несколько device:
     `remote_context.cpp:59` явно требует `devices.size()==1` — k независимых контекстов).
   - Клонировать модель на ранг; вызвать существующий путь компиляции per-rank.
     Пасс `TPShardTransform` в этом этапе — ЗАГЛУШКА/нет (шардирование — S06+); важно
     получить k графов.
   - Место для TP-пассов — рядом с плагинными пассами в `transformations_pipeline.cpp`
     (см. `gpu-plugin-arch.md` §3), до фьюжнов; в S04 достаточно точки расширения.
   - Собрать минимальный `TPPlan` (заготовка структуры: ранги, классификация швов пустая) —
     без коллективов.
3. **Обёртка** (заготовка, полное исполнение — S05):
   - `TPCompiledModel`/`TPInferRequest` — каркас (k графов, входы/выходы ранга 0), но
     корректный e2e-инференс не входит в GATE S04. Достаточно, чтобы compile вернул
     объект, способный сообщить `tp_ranks`.
4. **Тест/GATE**: мини-сеть с FullyConnected (ручная `ov::Model`: Parameter → MatMul/FC →
   Result), 2 GPU, `model_distribution_policy=TENSOR_PARALLEL` → компиляция проходит,
   `tp_ranks==[0,1]` (или корректный отчёт). Отрицательные кейсы: TP на 1 GPU,
   `"GPU.0"`+TP, `PIPELINE_PARALLEL` на GPU → внятная ошибка.
   Тест положить туда, где живут GPU-тесты (проверить дерево `src/plugins/intel_gpu/tests/**`
   и соглашения; не ломать существующие).

## GATE S04 (критерии завершения)
- TP-ветка `compile_model` компилирует 2-GPU мини-сеть; `TPCoordinator` создаёт k рангов.
- Свойства добавлены и читаются (`model_distribution_policy`, `enable_tensor_parallel`,
  `tp_ranks`); ошибки несовместимых комбинаций внятные.
- STATE.md: S04 done (или честный блокер, если сборка/тест невозможны в среде) +
  артефакты + решения + заметки → S05.
- NEXT.md — полный самодостаточный промпт S05 (Stage 1.2 TPInferRequest: реплика цельного
  графа на 2 GPU, выходы rank0; GATE — e2e паритет vs 1-GPU).
- Commit (сообщение: "S04: TPCoordinator (k engines, TP compile, props)") — **локально,
  без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений
  удалённого репозитория (см. STATE «Remote-политика»).
- Атомарность этапа: не реализовывать S05+ (реальный inference, шардирование, коллективы).
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если нужна сборка OpenVINO и она недоступна — не изобретать обходные сборки; зафиксировать
  блокер в STATE и считать GATE частичным (код + план теста готовы).
