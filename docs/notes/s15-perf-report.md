# S15: peer-копии + ring/tree allreduce: prefill/decode report

Канон: PLAN.md §4.4 (runtime/коллективы), §5 (S15), §6 (перф: `TPOT(TP1)/TPOT(TP2) ≥ 1.5`,
`TTFT(TP1)/TTFT(TP2) ≥ 1.3`), §7.4 (fp32-аккумуляция). Предшественники: `docs/notes/tp-bench-plan.md`
(S14, план бенча), `docs/notes/l0-spike.md` §3/§8 (G0: host-USM staging), `docs/notes/cpu-tp-pattern.md` §5.

## 0. Резюме

S15 добавляет в `TPTransport` два перф-механизма и отчётность:

1. **PEER-копии.** Проба `zeDeviceCanAccessPeer` теперь реальная (`cldnn::gpu_devices_peer_accessible`);
   при наличии peer-доступа транспорт переключается на выделенную ветку прямых device-копий
   (`TPTransport::PeerRegionCopy`). На заданной платформе (2× Arc A770, Windows) peer-доступ =
   `false`, поэтому ветка **подготовлена, но не активна** (используется host-staging, G0).
2. **Ring all-reduce.** `all_reduce_sum_fp32` выполняет ring reduce-scatter + all-gather вместо
   «все → rank0 → все»; fp32-аккумуляция сохранена, реплицированный bias/residual по-прежнему
   учитывается ровно один раз. Трафик/число сообщений доступны через `last_all_reduce_stats()`.

**GATE (TP2/TP1): частичный.** Бенч на реальном Qwen3.5-9B-int4 (2× A770 / 2× BMG) **не прогнан**:
среда сборки недоступна (сабмодули пусты, нет компилятора в PATH; см. §6). Ниже — аналитическая
модель (трафик + критический путь) и ожидаемый профиль prefill/decode; фактические TTFT/TPOT
снимаются по чек-листу §7, когда появится сборка/железо.

## 1. Что сделано

### 1.1. Проба peer-доступа
- `src/common/zero_loader/include/openvino/zero_api.hpp`: в `weak_symbols_list()` добавлен
  `zeDeviceCanAccessPeer` (слабый символ; отсутствует — не падаем).
- `src/plugins/intel_gpu/include/intel_gpu/runtime/device_query.hpp`: объявление
  `bool cldnn::gpu_devices_peer_accessible(size_t count)`.
- `src/plugins/intel_gpu/src/runtime/tp_peer.cpp` (новый): перечисляет L0-драйверы/устройства через
  `ov::ZeroApi`, для первых `count` Intel-GPU проверяет все упорядоченные пары `i≠j`, кэширует по
  `count`, **никогда не бросает**. Возвращает `false`, если устройств < 2, L0 недоступен или хотя бы
  одна пара без peer-доступа (консервативно — без ложных срабатываний).

### 1.2. Транспорт
- `tp_transport_capabilities()` → `tp_transport_capabilities(size_t num_ranks)`, результат кэшируется
  по числу рангов; `kind = PEER` только если `peer_available`. `TPTransport` теперь может
  конструироваться с уже известными capabilities (для тестов) либо с пробой по умолчанию.
- `PeerRegionCopy = std::function<bool(const Tensor&, const Tensor&, size_t axis, size_t offset,
  size_t count)>` + `set_peer_region_copy(...)`. В `all_gather_axis`/`scatter_axis` региональная
  копия идёт через бэкенд, только если `kind == PEER` **и** бэкенд установлен; иначе host-staging.
- `TPAllReduceAlgorithm {RING, HOST_SUM}` + `TPAllReduceStats {algorithm, messages, bytes}` +
  `last_all_reduce_stats()`.

### 1.3. Ring all-reduce
`ring_all_reduce_fp32(partials, out, messages, bytes)`:
- **reduce-scatter** (k−1 шагов): ранг `r` шлёт чанк `(r − step) mod k` преемнику `r+1` и
  накапливает пришедший от предшественника чанк; после k−1 шагов ранг `r` владеет **полностью
  редуцированным** чанком `(r+1) mod k`;
- **all-gather** (k−1 шагов): тот же кольцевой обмен чанками; в конце все ранги владеют суммой,
  возвращается буфер ранга 0.
- Требуется `S % k == 0`; иначе — `HOST_SUM` (прежний путь). Оба пути fp32.

Проверено вручную для k=2 и k=3 (индексы чанков reduce-scatter/all-gather согласованы, конфликтов
записи нет). Реплицированные терм-ы учитываются так же, как в S08: `acc -= (k−1)·term`.

## 2. Модель трафика и критического пути

Обозначения: `S` — число элементов в выходе GEMM (fp32), `k` — число рангов, `L` — число K-split
всех-редьюсов на шаг декодирования (слои × пункты).

| Схема | Трафик на ранг (bytes) | Крит. путь (bytes) | Сообщений |
|---|---|---|---|
| Было: все → rank0 → все | `(k−1)·S·4` отдача + `(k−1)·S·4` приём = `2(k−1)S·4` | `2(k−1)S·4` | `2(k−1)` |
| Ring (S15) | `2(k−1)/k·S·4` | `2(k−1)/k·S·4` | `2(k−1)·k` |

Для `k=2`: ring — **одно плечо в каждую сторону по S**, критический путь `S·4` байт против `2S·4`
у прежней схемы (вдвое меньше). Ring симметричен (нет «горячего» rank0), лучше масштабируется на
`k>2`, где host-sum деградирует линейно по числу рангов.

### 2.1. Пример (реальная форма Qwen, prefill)
Скрытая размерность ≈4096, `prompt_len=128`: выход одного K-split GEMM `S = 128·4096 = 524288`
элементов = **2 MiB fp32**. Ring k=2: ~2 MiB/ранг на коллектив против ~4 MiB ранее. На 36 слоях и
нескольких коллективах на слой это десятки–сотни MiB на prefill-проход через PCIe; при staging
полосе ~4–5 GB/s (G0) это уже заметное слагаемое (см. §3).

### 2.2. Важное ограничение
Данная модель покрывает только **K-split** коллективы (S08–S13). Реальный Qwen по-прежнему в
основном **REPLICATED** (coupled fused `in_proj`/`Concat`/`Loop`/`ScatterNDUpdate`; нет in-graph
коллективов). Значит ring-выигрыш применяется лишь к K-split-подграфам; полное покрытие графа —
задача будущей in-graph коллективной инфраструктуры (вне S15).

## 3. Prefill vs decode (ожидаемый профиль)

- **Prefill (TTFT).** Доминирует compute + первый размерный всплеск коллективов. Ring halving
  снижает долю staging на каждом K-split; при staging-floor ~60–90 µs/вызов (S03) выигрыш заметен
  на большом числе коллективов. Ожидается, что цель `TTFT ≥ 1.3×` достижима легче, чем TPOT.
- **Decode (TPOT).** Каждый шаг — `L` коллективов маленького `S` (batch=1, `S ≈ 4096` = 16 KiB).
  Здесь критичен **host-sync floor** (~60–90 µs/вызов), а не объём: ring сокращает число
  «остановок» вдвое для k=2, снимая ~половину staging-латентности на K-split-узлах. Однако пока
  REPLICATED-узлы доминируют, `wall ≈ max(rank) + staging`, и `TPOT_ratio ≈ 1.0`.
- **PEER.** При доступном peer-доступе прямые device-копии исключают pinned-host round-trip и
  host-sync floor — это главный рычаг к `TPOT ≥ 1.5×`; на A770 недоступен.

## 4. Ожидаемые узкие места (для будущего профиля)

1. REPLICATED-слои — TP2 исполняет полный граф на каждом ранге (S07–S13) → `TPOT_ratio ≈ 1.0`.
2. host-USM staging (G0) — линейно суммируется по слоям/шагам.
3. Serial-хвосты неполного параллелизма (см. S14 `m_rank_executor`).
4. Отсутствие peer-копий (A770) — нет пути вокруг host-sync floor.

## 5. Решение по GATE

- Код S15 реализован и покрыт функциональным тестом (§8); трафик/крит. путь и план замеров готовы.
- **GATE TPOT ≥ 1.5× / TTFT ≥ 1.3× не подтверждён измерением** — бенч не запускался (§6).
  Аналитически: для K-split-доминирующих графов ring halving даёт вклад в правильную сторону, но
  на реальном Qwen (преимущественно REPLICATED) ожидаемый `TPOT_ratio ≈ 1.0`, а `TTFT` — ниже
  целевого без in-graph коллективов и/или peer-копий.
- **Статус: частичный GATE.** Зафиксировано в STATE.md; передано в S16.

## 6. Блокер среды (почему бенч не прогнан)

- Сабмодули OpenVINO пусты; `build_tp/CMakeCache.txt` есть (Ninja, MSVC 14.44), но нет
  `build.ninja`/`compile_commands.json`; компилятора (`cl`) в PATH нет.
- Нет собранного плагина и `openvino_genai` → реальные TTFT/TPOT снять нельзя.
- Блокер не обходится «левыми» сборками; фиксируется в STATE.md.

## 7. Чек-лист запуска (когда доступна сборка/железо)

- [ ] Собрать OpenVINO из ветки, разложить Python-пакет (+ `openvino_genai`).
- [ ] `core.get_available_devices()` → `GPU.0`,`GPU.1`.
- [ ] Прогнать `tensor_parallel` функциональные тесты (S05–S15) — паритет и детерминизм ring.
- [ ] `python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov \
      --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json`.
- [ ] Снять профиль TP2 (`get_profiling_info`, префиксы `rankN:`): доля staging vs compute.
- [ ] При доступном peer-доступе (напр. 2× BMG с cross-device access) — зарегистрировать
      `set_peer_region_copy` и повторить замер (ожидается рост TPOT-ratio).
- [ ] Записать `TTFT_ratio`/`TPOT_ratio` в STATE.md → финальное решение GATE.

## 8. Тесты

- `KsplitRingAllReduceDeterministicAcrossRuns`
  (`src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp`): K-split с
  ALL_REDUCE_SUM-швом; повторный infer на том же запросе и свежий запрос дают **бит-в-бит**
  идентичный выход (детерминизм ring-порядка) и совпадают с 1-GPU (`cosine > 0.999`,
  `compare(1e-3)`).
