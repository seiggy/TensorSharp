# Independent encoder correctness oracle

[English guide](../../../embeddings.md) · [中文指南](../../../embeddings_zh-cn.md)

[`eng/embedding-reference.py`](../../../../eng/embedding-reference.py) computes
BERT/XLM-R embeddings directly from GGUF weights with NumPy. It uses `gguf-py`
only to read and dequantize weights. Token IDs come from independently saved
fixtures, and no TensorSharp or llama.cpp inference code participates in the
reference forward pass.

## Scope and acceptance criteria

Each model runs six inputs: two code-search queries, two Python functions, a
Chinese sentence, and a Unicode normalization example. Every token passes through
every encoder layer with FP32 projections, full per-sequence bidirectional
attention, layer normalization, exact erf GELU, model-declared pooling, and L2
normalization. The implementation materializes only the embedding rows used by
these inputs and dequantizes/frees one projection matrix at a time.

The gates, selected before the first run, are **cosine ≥ 0.999** and **maximum
absolute component error ≤ 0.005** for every input. Both code-query rankings must
also match the NumPy reference. The same gates apply to TensorSharp's pure C# CPU,
native GGML CPU, and Metal implementations, and to llama.cpp's CPU and Metal
implementations. These checks detect architecture and numerical mistakes;
six examples and two candidate documents do not establish MTEB or general
retrieval quality. The reference uses the same quantized GGUF weights, so it does
not measure quantization loss against the original full-precision checkpoint.

## Results

All six TensorSharp model/backend combinations pass both numerical gates for all
six inputs in both engine orders: 12 comparison reports, each containing six
vectors. Both retrieval rankings agree across every implementation/backend
comparison. Forward and reverse orders produce identical agreement metrics.

The `*-managed` files are pure C# `cpu`; `*-cpu` files are native `ggml_cpu`.
The table shows the worst agreement across both orders. Each row links its
forward report; reverse reports use the same filename with `-reverse` added.

| Model / backend | TensorSharp minimum cosine | TensorSharp maximum error | llama.cpp minimum cosine | llama.cpp maximum error |
|---|---:|---:|---:|---:|
| Snowflake / pure C# CPU | [0.999259271](snowflake-managed.json) | 0.004175864 | [0.999284980](llama-snowflake-managed.json) | 0.003939325 |
| Snowflake / Metal | [0.999986902](snowflake-metal.json) | 0.000609264 | [0.999986996](llama-snowflake-metal.json) | 0.000578269 |
| Snowflake / GGML CPU | [0.999288662](snowflake-cpu.json) | 0.003845549 | [0.999284980](llama-snowflake-cpu.json) | 0.003939325 |
| MiniLM / pure C# CPU | [0.999728062](minilm-managed.json) | 0.003653004 | [0.999711749](llama-minilm-managed.json) | **0.005061030** |
| MiniLM / Metal | [0.999998744](minilm-metal.json) | 0.000280410 | [0.999998760](llama-minilm-metal.json) | 0.000280410 |
| MiniLM / GGML CPU | [0.999746166](minilm-cpu.json) | 0.004049902 | [0.999711749](llama-minilm-cpu.json) | **0.005061030** |

The llama.cpp MiniLM CPU baseline exceeds the predeclared maximum-component
error gate by 0.000061030 on the Unicode example; its cosine and both retrieval
rankings pass. Its JSON correctly records `passed: false`. The threshold was
retained unchanged for both engines. The same baseline failure appears in both
engine orders alongside the managed and native TensorSharp CPU comparisons.
This small numerical fixture does not rank
the models' general retrieval quality.

Reverse reports: Snowflake [managed](snowflake-managed-reverse.json),
[GGML CPU](snowflake-cpu-reverse.json), [Metal](snowflake-metal-reverse.json);
MiniLM [managed](minilm-managed-reverse.json),
[GGML CPU](minilm-cpu-reverse.json), [Metal](minilm-metal-reverse.json).
Corresponding llama.cpp reports have the `llama-` filename prefix.

Each JSON records the model, token-fixture, HTTP-results, script, and reference
vector SHA-256 values. Reference `.npz` files also bind their vectors to the exact
selected token IDs and script that generated them.

## Reproduce

Run from the repository root, with the downloaded model files from
[`models.json`](../models.json) and an existing local llama.cpp checkout. The
recorded environment uses Python 3.13.15, NumPy 2.5.3, and Apple Accelerate.
NumPy 2.3.1 includes an [Accelerate floating-point warning
fix](https://numpy.org/doc/stable/release/2.3.1-notes.html); the earlier Python
3.9/NumPy 2.0.2 exploratory run was replaced with the newer environment.
Divide-by-zero, overflow, and invalid NumPy operations raise errors; every layer
and final response is additionally checked for finite values.

```bash
python3.13 -m venv /tmp/tensorsharp-numpy-oracle-venv
/tmp/tensorsharp-numpy-oracle-venv/bin/pip install numpy==2.5.3 pyyaml tqdm sentencepiece
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

/tmp/tensorsharp-numpy-oracle-venv/bin/python eng/embedding-reference.py \
  --gguf-py ../llama.cpp/gguf-py \
  --model ../models/embeddings/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  --tokens docs/validation/embeddings-2026-09/snowflake-tokenization.json \
  --http-results docs/validation/embeddings-2026-09/snowflake-metal/results.json \
  --output docs/validation/embeddings-2026-09/numpy-oracle/snowflake-metal.json

/tmp/tensorsharp-numpy-oracle-venv/bin/python eng/embedding-reference.py \
  --gguf-py ../llama.cpp/gguf-py \
  --model ../models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  --tokens docs/validation/embeddings-2026-09/numpy-minilm-tokens.json \
  --http-results docs/validation/embeddings-2026-09/minilm-metal/results.json \
  --output docs/validation/embeddings-2026-09/numpy-oracle/minilm-metal.json
```

The default fixture indices are `0,1,2,3,7,8`. Snowflake IDs are the saved
llama.cpp tokenizer fixture, independently cross-checked against HuggingFace
SentencePiece; MiniLM IDs come from the pinned HuggingFace tokenizer. See
[`huggingface-tokenization.json`](../huggingface-tokenization.json) for tokenizer
provenance and intentional Unicode compatibility differences.

To compare another HTTP run without repeating model inference, add
`--reference-vectors .../snowflake-metal.vectors.npz` (or the MiniLM file), change
`--http-results` and `--output`, and select `--http-engine tensorsharp` or
`--http-engine llama`. Managed CPU, native CPU, and Metal comparisons use the
same model's NumPy vectors. Final HTTP directories are
`{snowflake,minilm}-{managed,cpu,metal}` and the corresponding `-reverse`
directories. A failed gate writes its report and exits with status 1.
