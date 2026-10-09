# CODEFIX

CODEFIX is a small Python challenge project focused on implementing a KV-cached multi-head attention layer in PyTorch.

The goal is to build or fix a class named `KVCachedMultiHeadAttention` so it matches the expected behavior described by the included validator and test cases.

## Project purpose

This project is not a web application or library. It is a coding exercise designed to test:

- correct multi-head attention math
- correct Q/K/V projection logic
- proper cache concatenation behavior
- causal masking
- correct softmax dimension
- dropout only during training
- output and cache shape correctness

## Files

- `kv_attention.py` — the implementation under test
- `validator.py` — validates the implementation against test cases
- `test_cases.json` — visible validation cases
- `test_cases_hidden.json` — hidden-test template; this copy contains unresolved placeholders
- `attention_debugger.py` — optional debugging helper for checking common attention bugs
- `requirment.txt` — Python dependency list

## Requirements

Python 3.10+ and PyTorch are required.

Install dependencies:

```python -m pip install -r requirment.txt
```

## Run the validator

Use:

```python validator.py --file kv_attention.py --verbose
```

The included hidden-test template is not runnable until its placeholder values are replaced with actual tensors and expected outputs. If you have a populated hidden test file, run:

```python validator.py --file kv_attention.py --test-file test_cases_hidden.json
```

Run the static attention debugger with:

```python attention_debugger.py
```

## Expected behavior

The implementation should:

- accept `query`, `key`, and `value` tensors shaped like `(batch, seq_len, d_model)`
- project inputs into attention heads
- support cached keys and values from prior steps
- compute attention scores with appropriate scaling
- apply causal masking when enabled
- produce output shaped like the query tensor
- return updated cache tensors

The cache stores the projected keys and values in `(batch, sequence, d_model)` layout. Pass the returned cache back into the next call to continue incremental attention. Its sequence dimension may be at most `max_cache_len`.

## Notes

This challenge is essentially a bug-fix / implementation task in a PyTorch deep-learning setting. The validator checks whether the attention code is mathematically and structurally correct under a set of test cases.

## Summary

The project exists to evaluate whether a candidate can implement a correct KV-cached multi-head attention block from scratch, without relying on higher-level transformer libraries.
