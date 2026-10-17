# СЕССИЯ S07 — Stage 2.2 N-split + all-gather-эквивалент (срез весов до upload) · Проект TP-GPU: Tensor Parallel на GPU в OpenVINO

## Как работать
- Сначала полностью прочитай:
  1) C:\openvino-tp\PLAN.md — канон (НЕ редактировать в сессии), особенно §4.2 (compile-time, место
     TP-пасса и срез констант), §4.3 (N-split по головам/каналам, allgather-эквивалент), §5 (строка S07),
     §7 (риски: фьюжны через шов, срез до `propagate_constants`).
  2) C:\openvino-tp\STATE.md — живое состояние, особенно «Заметки сессия → следующая» блок **S06→S07**
     (что уже сделано: `TPShardTransform`, rt_info-транспорт, `tp_seams`), а также S04/S05
     (`TPCoordinator`, `TPCompiledModel`, `TPInferRequest`) и «Блокеры» (сборка недоступна).
  3) C:\openvino-tp\docs\notes\gpu-plugin-arch.md — §3 (где пасс и почему срез до `propagate_constants`).
  4) C:\openvino-tp\docs\notes\cpu-tp-pattern.md — §5–§6 (N-split FC, allgather = конкатенация без
     арифметики; численные инварианты).
- Правило проекта: в этой сессии выполняется РОВНО ОДИН этап = S07 (таблица PLAN.md §5). Следующие
  этапы (S08+ — K-split/allreduce, коллективы, per-rank KV) ЗАПРЕЩЕНЫ. Остановка на GATE/блокере —
  нормальный результат; зафиксируй его в STATE.md.

## Контекст, который УЖЕ исполнен (не повторять)
- S01–S03 done: ветка `feature/gpu-tensor-parallel`; L0-спайки (2× A770; peer/IPC NA). **G0 = host-USM
  staging** (dev→pinned host→dev, ~4–5 GB/s e2e) — транспорт для будущих коллективов; в S07 его ещё НЕ
  подключать (см. ограничения ниже).
- S04 done (частично): `TPCoordinator`/`TP-ветка compile_model`/свойства (`tp_ranks`,
  `enable_tensor_parallel`, `model_distribution_policy`).
- S05 done (частично): `TPInferRequest` (k per-rank sub-запросов, репликация входов, выходы rank-0,
  последовательный infer); `TPCompiledModel::create_*_infer_request`.
- **S06 done (частично):** identity/analysis пасс `TPShardTransform`
  (`src/plugins/intel_gpu/src/plugin/transformations/tp_shard_transform.{hpp,cpp}`):
  - `ov::pass::ModelPass`, возвращает `false`; регистрируется в `TransformationsPipeline::apply` **только
    при TP** (policy{TENSOR_PARALLEL} ИЛИ `enable_tensor_parallel`), сразу после
    `set_per_pass_validation(false)`, до `SDPAToVLSDPA`/фьюжнов/LowPrecision (и семантически до
    `propagate_constants` — PLAN §4.2/§7.2). MatMul ещё не сконвертирован в FC в этот момент.
  - Детектит `MatMul`/`FullyConnected`(`Compressed`), классифицирует: нет константных весов (операнд 1)
    → `REPLICA`; выход фида в `ov::op::v1::Add` → `K_SPLIT`(`ALL_REDUCE_SUM`); иначе → `N_SPLIT`
    (`ALL_GATHER`). Пишет `std::vector<TPSeam>` в `model->get_rt_info()["intel_gpu_tp_seams"]`.
  - `TPPlan`/`TPSeamKind{NONE,N_SPLIT,K_SPLIT,REPLICA}`/`TPCollectiveKind{NONE,ALL_GATHER,ALL_REDUCE_SUM}`
    (`include/intel_gpu/plugin/tp/tp_plan.hpp` + `src/plugin/tp/tp_plan.cpp`: `set_tp_seams`/`get_tp_seams`).
  - `TPCoordinator::compile_model_tensor_parallel` после `clone_and_transform_model` ранга-0 собирает
    `plan.seams = get_tp_seams(transformed_model)`.
  - RO-свойство `ov::intel_gpu::tp_seams` (`vector<string>` формат `KIND:COLLECTIVE:TYPE:NAME`).
  - Тесты S06 в `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`.
- **Среда сборки НЕДОСТУПНА**: тулчейн есть (cl 14.44/cmake/ninja), но **все git-сабмодули пусты**
  (`git submodule status` → все с `-`), cmake configure падает; инициализация = сетевой fetch (запрещён).
  См. STATE «Блокеры». Если сабмодули появились — собери GPU-плагин + тест `tensor_parallel`; иначе
  честный частичный GATE.
- **КРИТИЧНО (архитектурный факт S06):** сейчас `clone_and_transform_model` клонирует одну и ту же
  модель для КАЖДОГО ранга с одинаковым config → пасс не знает свой rank. Для реального среза в S07
  ранг нужно прокинуть в пасс (или в config per-rank). Реши это ДО среза (см. п.1).

## Задача S07 (полный объём этой сессии)
Stage 2.2 — **N-split + all-gather-эквивалент**: впервые реально менять граф. Для узлов, помеченных
`N_SPLIT`, срезать константные веса по выходным каналам (out-channels, N-ось) так, чтобы на устройство
каждого ранга уезжал только его слайс (срез — на OV-уровне, ДО `propagate_constants`/upload). Собрать
полный выходной буфер результата на каждом ранге без арифметики (all-gather = конкатенация слайсов).
K-split/allreduce и K_SPLIT-узлы в этой сессии НЕ реализовывать (S08).

1. **Прокидывание rank в пасс (обязательный предусловие).**
   - Вариант A (рекомендован): в `TPCoordinator::compile_model_tensor_parallel` перед
     `clone_and_transform_model` положить rank в per-rank `ExecutionConfig` (напр. новый internal-option
     `ov::intel_gpu::tp_rank` / или уже существующий механизм) и в `TransformationsPipeline::apply`
     передать/выставить его в `TPShardTransform(rank, num_ranks)`.
   - Вариант B: обернуть пасс в отдельный вызов после `clone_and_transform_model` по рангу (но потеряется
     позиция «до фьюжнов» — НЕ рекомендуем).
   - Зафиксируй выбор и мотивацию в STATE. Не ломай TP-off (`tp_rank` отсутствует → пасс не запускается).

2. **Срез констант (N-split, out-channels).**
   - В `TPShardTransform` (или новом `TPNsplitTransform`, регистрируемом там же) для каждого узла с
     `kind == N_SPLIT` и константным операндом-весами:
     - заменить Constant-веса на слайс по N-оси для своего ранга. Для `MatMul` с `transpose_b=false`
       веса `[K, N]` → срез по axis 1 (`[K, n_start:n_end]`); при `transpose_b=true` учитывать раскладку.
       Для `op::FullyConnected` (`transpose_b` атрибут) — аналогично.
     - срез делать **целыми N-группами** (кратно group_size для int4 — PLAN §2; на S07 тест fp32/f16, но
       хелпер сразу параметризовать по group-границам и падать с внятной ошибкой, если N не делится).
     - **bias/scales/zp**: bias (если есть) режется по той же оси вместе с весами; scales/zp на S07
       можно не трогать (int4 — S09), но не сломать их наличие.
     - **НЕ резать K-ось** и не трогать `K_SPLIT`/`REPLICA` узлы (S08).
   - Численный инвариант: результат каждого ранга — часть N-оси; для проверки в тесте можно взять модель,
     где N-split очевиден (MLP: hidden→FC (N=2*intermediate) → ... → FC).

3. **All-gather-эквивалент (сборка полного буфера без арифметики).**
   - На S07 достаточно **host-side**/логической сборки: после N-split каждый ранг считает свой слайс;
     для получения полного выхода потребителя (или model output) собрать слайсы рангов в правильном
     порядке (конкатенация по N-оси) — без суммирования. Минимальный вариант: собрать на rank-0
     (например, через существующий `TPInferRequest`, копируя выходы рангов в общий буфер). Реальный
     device-side/коллективный транспорт (host-USM staging, IPC-события) — S08/S12; на S07 допускается
     простая host-сборка, но она должна давать корректный полный результат.
   - ВАЖНО: контракт выходов S05 (выходы только rank-0) сохранить; если выход модели N-split-узла, то
     rank-0 должен отдать ПОЛНЫЙ тензор, а не свой слайс.
   - Если полная сборка на S07 не влезает в этап — честно ограничить сценарий тестом, где N-split узел
     имеет потребителя, не требующего полного N (напр. два независимых N-split FC с разными выходами;
     или проверять поэлементно по слайсам). Задокументировать компромисс в STATE.

4. **Не ломать S04/S05/S06.**
   - Паритет `TensorParallelInferMatchesSingleGpu` и все S04-негативы и S06-seams-тесты должны
     продолжать проходить (после N-split паритет станет нетривиальным — это и есть суть S07).
   - `TPPlan.seams` не деградировать; при срезе можно дополнить `TPSeam` полями (axis/группы) — тогда
     обнови `tp_seams`-строку и тесты.

5. **Тесты/GATE** (`src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp` или соседний
   файл того же каталога — ROOT-glob подхватит без правки CMake):
   - **Главный GATE:** FC/MLP-мини-сеть (Parameter → N-split FC → N-split consumer → Result) собирается
     в TP2, каждый ранг получает СВОЙ слайс весов (проверить через срез/собрать полный выход),
     полный результат совпадает с 1-GPU референсом с `cos > 0.999` (и abs/rel, где применимо). SKIP <2 GPU.
   - Негатив: N, не делящееся на число рангов → внятная ошибка (OPENVINO_THROW) или корректный
     fallback на REPLICA (решить и задокументировать).
   - TP-off регресс: обычная компиляция не срезает веса, `tp_seams == {}` при TP-off.
   - Опц.: детект/срез на Qwen LM (24 linear + 8 full attention) — если модель и сборка доступны;
     иначе частичный GATE.

6. **Регистрация**: новые `.cpp` под `src/plugin/**` подхватывает `file(GLOB_RECURSE PLUGIN_SOURCES
   src/plugin/*.cpp ...)` (`src/plugins/intel_gpu/CMakeLists.txt:147`); тест — ROOT-glob. Правок CMake
   быть не должно.

## GATE S07 (критерии завершения)
- N-split реализован: константные веса `N_SPLIT`-узлов срезаны по out-channels ДО upload (на OV-уровне,
  в TP-пассе до `propagate_constants`); на устройство ранга едет только его слайс.
- All-gather-эквивалент даёт корректный полный результат (host-сборка допустима на этом этапе).
- GATE-тест: FC/MLP-сеть паритет TP2 vs 1-GPU `cos > 0.999` (при недоступной сборке — код+план теста,
  честная пометка «частично»).
- `TPPlan`/`tp_seams` остаются стабильными; K_SPLIT/REPLICA узлы не тронуты; TP-off == upstream.
- STATE.md: S07 done/блокер + артефакты + решения (выбор механизма rank, способ сборки) + заметки → S08.
- NEXT.md — полный самодостаточный промпт S08 (K-split + allreduce: fp32-аккумуляция, bias в reducer,
  швы-события; GATE — паритет на K-split тесте).
- Commit (сообщение: "S07: N-split + all-gather equivalent (weight slicing before upload)") —
  **локально, без push**.

## Жёсткие ограничения
- **Только локально: remote (origin/upstream) НЕ трогаем** — никаких push/pull/изменений удалённого
  репозитория, включая `git submodule update` (сетевой fetch).
- Атомарность этапа: НЕ реализовывать S08+ (K-split/allreduce, device-side коллективы, per-rank KV,
  int4-специфичный срез S09). S07 = только N-split + all-gather-эквивалент.
- Не коммитить .bin/модельные файлы; секреты не писать в файлы/логи/историю.
- `PLAN.md` в сессии не редактировать; дельты — в STATE.md.
- Попутные находки — в STATE «Заметки сессия → следующая» или docs/notes/.
- Если нужна сборка OpenVINO и она недоступна (сабмодули пусты) — не изобретать обходные сборки;
  зафиксировать блокер в STATE и считать GATE частичным (код + план теста готовы).
