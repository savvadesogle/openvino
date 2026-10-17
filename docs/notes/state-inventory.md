# Qwen3.5-9B-int4 state inventory (S10)

Конспект S10. Проверено на `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml`
(IR version 11, tip upstream 85720fe2). Только чтение; `.bin` не коммитится.

## 1. Как представлен state в графе

Модель **stateful на уровне OpenVINO** (`ReadValue`/`Assign`, opset6). State — не
явные входы/выходы (не `Parameter`/`Result`) и не `Stateful` (`Stateful` в IR v11
отсутствует: 0 совпадений). Переменные регистрируются через
`ov::op::util::Variable`; `ov::Model::get_variables()` / `get_sinks()` их видят
(в GPU-плагине они становятся `network->get_variables_info()` и попадают в
`InferRequest::query_state()`).

Счётчики по XML:
- `ReadValue` = 64, `Assign` = 64, `Variable` (объявления) = 64.
- `ScaledDotProductAttention` = 8 (full attention), `GroupConvolution`/`Loop`/
  `ScatterUpdate` = 24 каждый (linear attention).

`TPShardTransform` вставлен (`transformations_pipeline.cpp:660`) до любых
state-оптимизаций; на его входе state — «сырые» `ReadValue`/`Assign` с
`variable_id` и `variable_shape` из IR.

## 2. Формы переменных (фактические)

Все переменные `f32`, первая ось (batch) — динамическая.

| Тип | `variable_id` (шаблон) | `variable_shape` | Кол-во | Слой | Смысл осей |
|---|---|---|---|---|---|
| KV key   | `cache_params.past.key.Kcache_params.present.key.K`   | `?,4,?,256` | 8 | full_attention (счётчик K=0..7) | `[batch, kv_heads=4, seq, head_dim=256]` |
| KV value | `cache_params.past.value.Kcache_params.present.value.K` | `?,4,?,256` | 8 | full_attention | `[batch, kv_heads=4, seq, head_dim=256]` |
| conv     | `cache_params.past.conv.Ncache_params.present.conv.N`  | `?,8192,4`  | 24 | linear_attention (N=0..23) | `[batch, conv_channels=8192, kernel=4]` (depthwise conv state) |
| ssm      | `cache_params.past.ssm.Ncache_params.present.ssm.N`    | `?,32,128,128` | 24 | linear_attention | `[batch, la_heads=32, key=128, value=128]` (delta-rule recurrent state) |

Итого 64 переменные = 8 key + 8 value + 24 conv + 24 ssm.

### Оси head-размерности
- KV: ось **1** — число kv-голов (4); последняя ось — `head_dim` (256).
  GQA: 16 q-голов / 4 kv-головы (ratio = 4). `head_dim=256`.
- ssm: ось **1** — число linear-attention голов (32); оси 2,3 — key/value (128×128).
- conv: ось **1** — 8192 каналов (не голова в смысле GQA; это depthwise-канал).

Для 2 (и 4) рангов: 4 kv-головы / 2 = 2, /4 = 1; 16 q-голов /2 = 8, /4 = 4;
32 la-голов /2 = 16, /4 = 8. Все head-размеры делятся на число рангов.

## 3. Как это связано с проекциями (PLAN §4.3)

- `full_attention`: `in_proj_qkv` (N=8192 = (16+4+4)*256) → reshape по головам →
  RoPE → KV `Concat`/`ReadValue`/`Assign` → `ScaledDotProductAttention` →
  `o_proj`. KV-кэш обслуживает `ScaledDotProductAttention`.
- `linear_attention`: `in_proj_qkv` (N=8192) → depthwise `GroupConvolution`
  (state=`conv.N`, ядро 4) → delta-rule `Loop`/`ScatterUpdate` (state=`ssm.N`) →
  `out_proj`. Head-размерность ssm — ось 1 (32 головы).

## 4. Решение S10 по шардингу state

Полноценный per-rank KV (головы на разных рангах) требует, чтобы
`qkv → reshape → RoPE → SDPA → o_proj` и `in_proj → conv/Loop → out_proj` были
согласованно порезаны по головам и включали **межранговый all-gather** (полный
вход следующего узла) — это S12/S10+ (device-side коллективы). На S10:

- **Фактический Qwen → REPLICATED.** Причины (обе):
  1. `variable_shape` динамический (`?,4,?,256`, `?,8192,4`, `?,32,128,128`);
     статический head-срез невозможен без динамических шейпов (S12).
  2. Цепочки state не являются head-local: `ScaledDotProductAttention`,
     `GroupConvolution`, `Loop`, `ScatterUpdate`, `Concat` — не element-wise;
     срез одной лишь переменной рассогласовал бы шейпы.
- **Механизм sharding всё равно реализован** (S10) для head-local состояния:
  `TPShardTransform` распознаёт переменную со статическим head-размером,
  делящимся на число рангов, и переводит её в `HEAD_SHARDED`, только если вся
  цепочка `ReadValue → (elementwise/broadcast) → Assign` head-local (нет
  не-elementwise потребителей, все прочие операнды broadcast'ятся по head-оси).
  Тогда шейп переменной уменьшается до `[.., H/ranks, ..]` per-rank, и
  `TPInferRequest` агрегирует per-rank state конкатенацией по head-оси
  (`query_state`/`get_state`), а `set_state`/`reset` распределяет по рангам.
- **linear_attention (conv/ssm)** — S11 отдельной задачей (delta-состояние,
  conv). На S10 остаётся REPLICATED и агрегируется как rank-0 копия.

## 5. Ограничения/заметки
- В IR v11 нет `Stateful`; важно не путать с opset1 `Stateful` (устаревший).
- `ReadValue`/`Assign` (v6) шарят `ov::op::util::Variable`; изменение
  `data_shape` переменной propagate'ится на выход `ReadValue` при
  `validate_nodes_and_infer_types()`.
- `Assign` — `Sink` (нет output-потребителей): его вход должен быть достижим из
  `ReadValue` той же переменной (чистая рекуррентность) для head-среза.
