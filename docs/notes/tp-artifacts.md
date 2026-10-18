# TP-GPU: карта артефактов, ограничения и воспроизведение (S16)

Канон: PLAN.md §4–§7. Итоговый обзор проекта GPU tensor-parallel (TP) в плагине `intel_gpu`:
где лежит код, что работает, чего проект **сознательно не делает**, и как воспроизвести проверку
на машине со сборкой и 2+ Intel GPU. Сводка по перфу — `docs/notes/s16-perf-matrix.md`;
план выноса в upstream — `docs/notes/upstream-series.md`.

## 1. Что такое TP-режим (кратко)

Пользователь компилирует модель на устройство `"GPU"` с
`ov::hint::model_distribution_policy{TENSOR_PARALLEL}` (или internal `ov::intel_gpu::enable_tensor_parallel`).
Плагин компилирует модель **один раз на каждый GPU** (per-rank), срезая веса `MatMul`/`FullyConnected`
между рангами, и на инференсе собирает полный выход на ранге 0 (all-gather/all-reduce через
host-USM staging). Публично это один `ov::CompiledModel`.

## 2. Карта кода

| Слой | Файл | Роль |
|---|---|---|
| Вход/маршрутизация | `src/plugins/intel_gpu/src/plugin/plugin.cpp` (`compile_model`, `import_model`, `import_model_tensor_parallel`, `compile_model_tensor_parallel`), `plugin.hpp` | ветка TP, per-rank blob-import |
| Координатор | `src/plugins/intel_gpu/src/plugin/tp/tp_coordinator.{hpp,cpp}` | перечисление рангов, `TPlan`, компиляция per-rank |
| План/сечения | `include/intel_gpu/plugin/tp/tp_plan.hpp`, `src/plugin/tp/tp_plan.cpp` | `TPPlan`/`TPSeam`/`TPOutputSeam`/`TPStateSeam`, rt_info transport, `save_tp_plan`/`load_tp_plan` |
| Пас шардинга | `src/plugins/intel_gpu/src/plugin/transformations/tp_shard_transform.{hpp,cpp}` | детект/классификация `MatMul`/FC, срез весов N/K-split, int4-ASYM dequant, state-шардинг |
| Compiled model | `src/plugins/intel_gpu/src/plugin/tp/tp_compiled_model.{hpp,cpp}` | ранги, синтетические порты, scheduler, `export_model` (конверт) |
| Инференс | `src/plugins/intel_gpu/src/plugin/tp/tp_infer_request.{hpp,cpp}` | k sub-запросов, репликация входов, сборка выходов, агрегация состояния, scheduler |
| Транспорт | `src/plugins/intel_gpu/src/plugin/tp/tp_transport.{hpp,cpp}` | host-USM staging, all-gather/scatter/all-reduce, ring all-reduce, peer fast-path |
| Peer-probe | `src/plugins/intel_gpu/src/runtime/tp_peer.cpp`, `include/intel_gpu/runtime/device_query.hpp`, `src/common/zero_loader/include/openvino/zero_api.hpp` | `zeDeviceCanAccessPeer` пробник (не бросает) |
| Свойства | `include/intel_gpu/runtime/internal_properties.hpp`, `src/plugin/options.inl` | `tp_ranks`, `tp_seams`, `tp_output_seams`, `tp_state_seams`, `tp_rank`, `tp_num_ranks`, `enable_tensor_parallel` |
| Тесты | `src/plugins/intel_gpu/tests/functional/behavior/tensor_parallel.cpp` | GATE-тесты S05–S16 |
| Бенч | `tools/tp_bench/tp_bench.py`, `tools/tp_bench/README.md`, `docs/notes/tp-bench-plan.md` | TTFT/TPOT TP1 vs TP2 vs MULTI |
| Прочие доки | `docs/notes/{gpu-plugin-arch,cpu-tp-pattern,state-inventory,genai-integration,int4-asym-weights,l0-spike,l0-spike-plan,s15-perf-report,s16-perf-matrix,upstream-series}.md` | разведка/архитектура/отчёты |

## 3. Поддерживаемые возможности

- **N-split** (`TPSeamKind::N_SPLIT`): срез по выходным каналам, сборка **concat** (all-gather),
  полный синтетический порт на ранге 0.
- **K-split** (`K_SPLIT`): срез весов и активаций по оси сокращения, сборка **all-reduce(sum)** fp32;
  реплицированные bias/residual (`TPReplicatedAdd` CONSTANT/INPUT) учитываются ровно один раз.
- **int4_ASYM** (`u4` + scale/zp dequant-цепочка): срез weight/scale/zp по N-чанкам (N-split) или по
  group-оси (K-split) с выравниванием по байту; невыровненный/неделимый случай → REPLICA.
- **State** (`TPStateSeam`): head-sharded (KV/conv/ssm-подобные head-local цепочки) с агрегацией
  `query_state()/set_state()`; остальное — REPLICATED (авторитет ранга 0).
- **Scheduler**: k рангов исполняются параллельно на общем `IStreamsExecutor` (по stream на ранг).
- **Per-rank blob-cache** (S16): export/import конверта с независимым single-device блобом на ранг.
- **GenAI-совместимость**: строковый конфиг `{"MODEL_DISTRIBUTION_POLICY":"TENSOR_PARALLEL"}`, name-based
  `set_tensor/get_tensor`, `query_state/set_state`.

## 4. Ограничения (осознанные, вне S16)

1. **Tick модель Qwen3.5-9B-int4 остаётся преимущественно REPLICATED.** Реальные KV/conv/ssm-цепочки
   и coupled fused `in_proj`/`Concat`/`Loop`/`ScatterNDUpdate` не head-local → шардинг небезопасен;
   per-rank шардинг реальных слоёв требует **in-graph** межранговых коллективов (не реализовано).
2. **Коллективы — host-USM staging (G0)** через pinned-хост. На целевой платформе (2× Arc A770,
   Windows) `zeDeviceCanAccessPeer` = false → peer-копии подготовлены, но неактивны. На 2× BMG с
   cross-device access — зарегистрировать `TPTransport::set_peer_region_copy`.
3. **Нет device-side событий / zero-copy rank-импорта**: rank-модели кэш-блоба пересобираются из памяти
   (не mmap). Модель/размеры корректны, оптимизация mmap — возможная доработка.
4. **Границы среза** требуют делимости: `N % num_ranks == 0` (N-split), `K % num_ranks == 0` (K-split),
   `G % num_ranks == 0` (int4 K-split по группам), head-ось состояния статична и делима. Иначе — REPLICA
   (без throw, классификация seam сохраняется).
5. **Numeric**: вывод собирается в fp32 (PLAN §7.4), затем кастуется к типу выхода модели.
6. **Только 2+ физических Intel GPU**; TP нельзя совмещать с явным `GPU.N`/remote-context/pipeline-parallel.

## 5. Как воспроизвести (машина со сборкой + 2× Intel GPU)

```powershell
# 1) Сборка плагина (см. CONTRIBUTING.md), сборка функциональных тестов GPU.
#    Перед использованием подтянуть сабмодули (на целевой машине сеть разрешена):
git submodule update --init --recursive

# 2) Проверить, что видно 2 GPU.
python -c "import openvino as ov; print(ov.Core().get_available_devices())"   # -> ['CPU','GPU.0','GPU.1']

# 3) Функциональные GATE-тесты TP (паритет/детерминизм/кеш/состояние).
#    (путь бинаря зависит от вашей сборки)
.\bin\intel64\Release\ov_intel_gpu_behavior_tests.exe --gtest_filter=*TensorParallel*

# 4) Бенч TP1 vs TP2 vs MULTI (нужен openvino_genai для реальных TTFT/TPOT).
python tools/tp_bench/tp_bench.py --model-dir models/Qwen3.5-9B-int4-ov `
    --max-new-tokens 128 --warmup 2 --iters 10 --variants TP1,TP2,MULTI --out tp_bench_report.json
```

Ожидаемые результаты (см. `docs/notes/s16-perf-matrix.md`): паритет TP2 vs 1-GPU — да; ускорение
`TPOT ≥ 1.5×` — **не** на текущем Qwen (REPLICATED), требуется in-graph коллективы и/или peer.

## 6. Статус

Все стадии S04–S16 реализованы в коде и покрыты тестами, но **GATE по сборке/прогону не подтверждён**
в этой среде (сабмодули пусты, нет 2 GPU/GenAI). Подробности — STATE.md (журнал гейтов, блокеры).
