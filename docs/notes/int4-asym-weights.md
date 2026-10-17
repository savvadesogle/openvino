# INT4_ASYM (group-quantized) weights in the OV graph — findings for S09

Конспект S09. Проверено на `models/Qwen3.5-9B-int4-ov/openvino_language_model.xml`
(IR version 11, tip upstream 85720fe2). Только чтение; `.bin` не коммитится.

## 1. Как выглядят int4-веса в графе на момент `TPShardTransform`

Модель на диске — «сырой» OV IR. Линейные проекции — это `MatMul` (не
`FullyConnected`/`FullyConnectedCompressed`): FC-конверсия
(`ConvertMatMulToFullyConnected` → `ConvertFullyConnectedToFullyConnectedCompressed`)
идёт на `transformations_pipeline.cpp:1758/1763`, т.е. **сильно позже** пасса
`TPShardTransform` (`:660`). `MarkDequantization` (`:701`) тоже идёт после пасса.

Пример (layer 0, `linear_attn.in_proj_qkv`, N=8192, K=4096, gs=128, G=32):

```
weight  Const u4   [8192, 32, 128]          # [N, G, gs]
        -> Convert(f16)
zp      Const u4   [8192, 32, 1]            # [N, G, 1]
        -> Convert(f16)
        -> Subtract(Convert(weight), Convert(zp))
scale   Const f16  [8192, 32, 1]            # [N, G, 1]
        -> Multiply(sub, scale)               # [8192, 32, 128]
        -> Reshape([8192, 4096])              # [N, K]
        -> Convert(f32)
        -> MatMul(A, B, transpose_b=true)     # out [.., 8192]
```

⇒ На входе `MatMul` (операнд 1) стоит **не «голый» Constant**, а dequant-подграф.
Поэтому `has_constant_weights()` (S06) для int4-проекций возвращал `false` и
узел классифицировался как `REPLICA` — int4 на S06–S08 реплицировался (как и
задокументировано в STATE).

## 2. Представление, выбранное для S09

Стратегия **(i)**: `TPShardTransform` сам распознаёт линейную dequant-цепочку
(`Convert/Reshape/Multiply/Subtract` над `Constant`) и режет её компоненты
(`weight` + `scale` + `zero_point`, и `bias`-`Add`-константу для N-split).
`FullyConnectedCompressed` (стратегия ii) недоступен: он материализуется позже
пасса; на S09 срез на cldnn-уровне запрещён.

Распознавание (совпадает с `compressed_weights_pattern.hpp` и
`ConvertFullyConnectedToFullyConnectedCompressed`):
- walk назад от `gemm->input_value(1)` через `Convert`/`Reshape`/`Multiply`/
  `Subtract`;
- первый целочисленный `Constant` битности ≤ 8 — **weight**;
- константа на «прочей» ветви `Multiply` — **scale**;
- константа (возможно через `Convert`) на «прочей» ветви `Subtract` — **zp**;
- layout: weight `[.., N, G, gs]`, scale/zp `[.., N, G, 1]` (последний dim у
  scale/zp часто 1); `K = G * gs`.

## 3. Group-границы и sub-byte упаковка

- **N-split**: срез по оси `N` (обычно ось 0) целыми out-channel'ами;
  `N % ranks == 0`. groups лежат по K (`G`), N-срез их не трогает.
- **K-split**: срез по оси `G` (предпоследняя) целыми группами;
  требование `G % ranks == 0` (⇔ `K % (ranks * gs) == 0`, при gs=128 и
  K=4096, ranks=2 → OK). `scale`/`zp` режутся той же осью `G`; `Reshape`-цель
  `[N, K]` → `[N, G/ranks * gs]`.
- **zp с одним элементом (per-tensor) НЕ режется** (как CPU,
  cpu-tp-pattern.md §5).
- **sub-byte упаковка** (u4/i4): OV пакует 2 нибла на байт по row-major. Срез
  допустим, только если выровнены **три** величины (правило реализовано в
  `slice_is_byte_aligned`; важно — именно `row_bits`, а не `inner_bits`):
  `row_bits % 8 == 0 && off_bits % 8 == 0 && slab_bits % 8 == 0`, где
  `row_bits = shape[axis] * inner * bitwidth`, `off_bits = offset * inner * bitwidth`,
  `slab_bits = count * inner * bitwidth`, `inner` = произведение осей ПОСЛЕ `axis`.
  Для `[N,G,gs=128]`: по оси N inner=G*gs, по оси G inner=gs — выровнено при
  любых offset. Для `[N,G,1]` по оси G inner=1 → нужны `offset*4 % 8 == 0` и
  `count*4 % 8 == 0` (чётные `offset_g`/`count_g`; при gs=128 выполняется).
  Невыровненный срез → fallback в `REPLICA` (без throw).

## 4. Ограничения (наследуются/документируются)

- Реальные Qwen `qkv` (→rope/reshape/sdpa) и MLP `gate/up` (→elementwise→
  `down_proj` GEMM) требуют **настоящего all-gather между узлами** (полный N на
  входе следующего GEMM) — вне S09. Строгий check «elementwise-цепочка не
  доходит до другого GEMM» переводит такие узлы в `REPLICA` (без throw).
- K-split `o_proj`/`down_proj` имеют внутренний residual (не `Parameter`/
  `Constant`) → `scan_k_reduction` даёт `safe=false` → `REPLICA` (ограничение S08).
- Полноценный GATE на Qwen3.5-9B (TP2 prefill) недостижим до S10+ (per-rank KV,
  inter-node all-gather). S09 GATE — int4-мини-модель (см. тесты).
