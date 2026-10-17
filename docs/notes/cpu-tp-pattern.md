# CPU TENSOR_PARALLEL — паттерн реализации (эталон для GPU-порта)

Конспект S01. Все пути/строки проверены на master `85720fe2` (2026-10-17).
Будущие сессии НЕ пересобирают эти знания — только дополняют при изменениях upstream.

## 1. Включение и API

- Публичный API: `src/inference/include/openvino/runtime/properties.hpp:436-442`
  (`ModelDistributionPolicy::{TENSOR_PARALLEL, PIPELINE_PARALLEL}`),
  property `MODEL_DISTRIBUTION_POLICY` (`:488`), докстринг упоминает
  "between CPU sockets/NUMA nodes or between different GPUs" (`:470-486`).
- CPU принимает только TENSOR_PARALLEL: `src/plugins/intel_cpu/src/config.cpp:152-170`
  (иначе `OPENVINO_THROW`).
- Внутренний тест-флаг (RW, bool): `ov::intel_cpu::enable_tensor_parallel` —
  `src/plugins/intel_cpu/src/internal_properties.hpp:70-74`; читается из
  property в `plugin.cpp:680-681`, отдаётся в get_property
  `compiled_model.cpp:388-391`.
- Флаг передаётся в расчёт стримов: `cpu_streams_calculation.cpp:1421`
  (`config.enableTensorParallel`), проверка политики `:590-593`.

## 2. Стримы и ранги

- TP-ранжирование живёт в `src/plugins/intel_cpu/src/cpu_streams_calculation.cpp`:
  - latency + TP (одна подмодель на сокет): `handle_latency_tensor_parallel_or_single_socket()`
    `:619-657`; multi-socket TP: `populate_table_tensor_parallel()` `:822-862`
    (по сокету — по одному стриму с `SUB_STREAMS_FOR_SOCKET`).
  - single-socket, latency: `:966-973` — в `streams_info_table` добавляются
    2 «фантомных» ряда с `NUMBER_OF_STREAMS = -1` (маркер под-стримов),
    у ряда 0 `THREADS_PER_STREAM` удваивается.
  - multi-socket TP: `:1015-1021`.
  - `get_streams_rank_table()` `:1092-1117`: для каждого ряда с
    `NUMBER_OF_STREAMS<0` добавляется `|n|` sub-стримов; `num_sub_streams += |n|`;
    `rank = индекс sub-стрима` (при `rank_level==2` половина получает rank[0]=1).
  - `config.streamsRankTable = get_streams_rank_table(...)` — только при наличии
    TENSOR_PARALLEL в политике: `:1428-1432`.
- Итог: **число рангов == число sub-стримов == numSubStreams**; ранг — просто
  индекс; ранги не привязаны к физическим ядрам жёстко, это координата для
  шардинга данных.

## 3. CompiledModel: N подмоделей на один model

`src/plugins/intel_cpu/src/compiled_model.cpp:152-177`:

- Условие: `m_cfg.numSubStreams > 0` (ставится только TP-путём).
- Создаётся общий `SubMemoryManager(numSubStreams)` (`:159`) и
  `message_manager()->set_num_sub_streams()` (`:160`).
- N sub-CompiledModel, у каждого суб-конфиг с `numSubStreams=0`,
  `enableNodeSplit=true`, `streamsRankTable[i]` (их ранг) (`:161-176`).
- `create_infer_request()` `:239-256`: создаются N sub-запросов и через
  `async_infer_request->setSubInferRequest(requests)` + `setSubInfer(true)`
  (`:246-254`).

## 4. InferRequest: параллельный запуск рангов

- `src/plugins/intel_cpu/src/infer_request.cpp`:
  - `SyncInferRequest::infer()` `:103-114`: если `m_has_sub_infers` →
    `sub_streams_infer()` + `message->server_wait()` (ждём N callback'ов).
  - `sub_streams_infer()` `:633-662`: **выходы биндятся только на rank 0**
    (`requests[0]->set_tensor(output, ...)`, `:643-646`); **входы — на все
    ранги** (`:647-651`); каждому ставится callback → `MessageManager`;
    затем все N стартуют `start_async()` (`:658-660`). Ранги исполняются
    реально параллельно; синхронизация — поузловая (см. §5), не глобальная.
  - `query_state()` `:317-328`: состояния всех рангов конкатенируются.
- `async_infer_request.cpp:37-55` — только обвязка sub-запросов.

## 5. Механика шва: FC + SubMemoryManager (allgather)

`src/plugins/intel_cpu/src/sub_memory_manager.hpp:12-48`:
- `MemoryInfo{ send_buf, flag, last_used }`; таблица `_memorys_table[2][n_sub]` —
  двойная буферизация (переключение `last_used` в `set_memory_used` `:39-42`);
- `get_memory_id()` `:30-37` выбирает свободный буфер;
- `_flagMutex` и `_use_count` — примитивы синхронизации рангов.

`src/plugins/intel_cpu/src/nodes/fullyconnected.cpp` (int4/FC-узел):
- `initTensorParallelConfig()` `:241-251`: `w_rank = executor->get_rank()[0]`,
  `w_size = message_manager()->get_num_sub_streams()`,
  `enable_tensor_parallel = w_size > 1`, `sub_memory = context->getSubMemory()`.
- Динамика шейпов: `needPrepareParamsForTensorParallel()` `:294-324` — dst
  последней dim режется `split_parts(len,n)` и каждый ранг использует свою
  часть (`cloneWithNewDims`); проверка `dims[-1] >= w_size`.
- Перед исполнением — спин-барьер `initTensorParallelSync()` `:337-355`:
  ждём `_use_count[id]==0` под `_flagMutex`; при `== w_size` счётчик и флаги
  сбрасываются (последний пришедший «очищает» раунд).
- Исполнение `execute()` `:448-458`: барьер → локальный GEMM на своём слайсе →
  **allgather** `execTensorParallelSync()` `:357-446`:
  1) публикация результата ранга: `send_buf = cur_dst->getData(); flag = true`
     (`:389-390`);
  2) спин-ожидание флагов всех рангов (`:392-440`);
  3) сборка полного dst: для каждого ранга `idx` — `parallel_for`-memcpy
     кусками по `splited_dim_vec[idx]` байт в позицию `idx * strideSize` по
     последней оси (`:401-432`) — **конкатенация, без арифметики**;
  4) `_use_count[id]++` под мьютексом (`:441-444`).
- Слайсинг памяти: `needSplitMemoryForTensorParallel()` `:660-719`:
  - веса: N-split по out-channels — `split_vertical(dim0)` для
    `weightsNonTransposed`, иначе `split_horizontal(dim0)` (`:669-673`);
  - bias: `split_horizontal(dim0)` (`:674-682`);
  - dst: `split_horizontal(dim=-1, need_fill=false)` — пустая память-вид без
    копии (`:685`);
  - **scales и zero_points режутся тем же способом, что и веса** (`:687-717`);
    zp с 1 элементом не режется (`:703-716`).
- Предохранители `needUpdateTensorParalelConfig()` `:721-737`: TP выключается,
  если шейп весов динамический, либо `dims[0] < w_size`, либо это 3D-веса с
  размером последней оси > 1.
- Порядок в `createPrimitive()` `:739-756`: updateCfg → заполнить `memory` из
  портов → `needSplitMemoryForTensorParallel()` → `factory->make(memory)`.

Помощники слайсинга: `src/plugins/intel_cpu/src/cpu_memory.cpp`
- `split_horizontal()` `:630-691` — **вид без копирования**: новый desc + ptr
  `srcPtr + w_rank * stride` (для u4/i4 stride/2); динамический dim остаётся
  динамическим (`:649-663`), `need_fill=false` → пустой буфер.
- `split_vertical()` `:693-761` — копирование столбцов своего ранга
  (`dims[dim]*prec.size()` на строку), поддержка динамических шейпов,
  `need_fill=false` → только desc.

## 6. Выводы для GPU-порта

1. **Ранг = индекс подмодели**, вкомпилированный в граф на compile-time;
   per-rank Engine/Context + свой слайс констант (у CPU — view на общий буфер,
   у GPU — upload только слайса, PLAN §4.2).
2. **Синхронизация**: у CPU — host-spin по флагам в общей памяти после каждого
   FC-узла (только внутри одного процесса/памяти). Для GPU нужны
   device-side wait'ы: IPC-события + publish в staging (PLAN §4.4); структуру
   «двойного буфера + счётчик раунда» (`SubMemoryManager`) можно переиспользовать
   как идею пула staging-буферов по (seam, rank, iteration).
3. **Allgather = конкатенация** N-слайсов в полный буфер (без арифметики);
   на GPU: publish-копия в staging + assemble-ядро (или P2P-copy), fp32/bf16
   без потерь.
4. **K-split/allreduce на CPU НЕТ вообще** — CPU FC режет только N
   (out-channels); механику allreduce(sum) для GPU (O-proj, out-proj
   linear-attention) проектируем с нуля (PLAN §4.3): fp32-аккумуляция,
   bias добавляется один раз в reducer.
5. **Численные инварианты**: CPU `split_parts` делит «поровну с хвостом на
   последнем ранге» (неравные хвосты!). Для int4_ASYM это ломает group-границы
   (group_size=128) — GPU-шардер обязан резать K кратно 128 и N целыми
   oc-группами (PLAN §2); `split_parts` годится только как референс механики.
6. **Контракт запроса**: входы реплицируются на все ранги, выходы биндятся
   только к рангу 0, состояния агрегируются — этот контракт копируем в
   TPInferRequest (PLAN §4.4).
7. `enable_tensor_parallel` (internal bool) — удобный паттерн «тестовый флаг»
   отдельно от публичной политики: у GPU завести аналог
   (`ov::intel_gpu::enable_tensor_parallel`, PLAN §4.1).
