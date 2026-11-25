# SMS scam detector models (final, trained 19 Nov)

Trained the same way as the honest cross-validation (ml.cv_real): 463 real Indian SMS + their synthetic rewordings + 20% of public data. Honest accuracy (cross-validated, deep + rules): 90.9%. These files trained on all 463 real messages, so do not score them on those messages.

App cut-offs on the combined risk score (deep model): Suspicious >= 0.39, Scam >= 0.92.

| File | Size (KB) | Params | Max diff vs PyTorch | Max diff vs fp32 |
|---|---|---|---|---|
| `sms_tiny_fp32.pte` | 58.1 | 13,473 | 4.02e-07 | 0.000 |
| `sms_tiny_int8.pte` | 31.6 | 13,473 | 5.96e-08 | 0.019 |
| `sms_deep_fp32.pte` | 611.3 | 154,177 | 4.17e-07 | 0.000 |
| `sms_deep_int8.pte` | 187.0 | 154,177 | 5.96e-08 | 0.006 |

## Interface

- Runtime: ExecuTorch, XNNPACK backend, method `forward`
- Input: `int64` tensor `[1, 200]`, character ids from `vocab.json` (this export's vocab only)
- Output: `float32` tensor `[1]`, scam probability in `[0, 1]`
- Reference inputs and expected outputs: `test_vectors.json`

## Operators

- `sms_tiny_fp32.pte`: 3 XNNPACK delegate block(s); outside XNNPACK: alloc x4, embedding.out x1, amax.out x1, cat.out x1, squeeze_copy.dims_out x1
- `sms_tiny_int8.pte`: 4 XNNPACK delegate block(s); outside XNNPACK: alloc x4, embedding.out x1, amax.out x1, cat.out x1, squeeze_copy.dims_out x1
- `sms_deep_fp32.pte`: 3 XNNPACK delegate block(s); outside XNNPACK: alloc x5, embedding.out x1, amax.out x3, squeeze_copy.dims_out x1
- `sms_deep_int8.pte`: 3 XNNPACK delegate block(s); outside XNNPACK: alloc x5, embedding.out x1, amax.out x3, squeeze_copy.dims_out x1
