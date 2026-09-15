# Embedding validation — September 2026

This report covers the source implementation added for
[discussion #183](https://github.com/zhongkaifu/TensorSharp/discussions/183).
See [usage and API contracts](../../embeddings.md),
[pinned models and checksums](models.json), and the
[HTTP benchmark runner](../../../benchmarks/EmbeddingBench/README.md).

## Environment and method

- Apple M5 Pro, 18 CPU cores, 48 GiB unified memory; macOS 26.6 arm64.
- .NET SDK 10.0.302, runtime 10.0.10; Release managed and native builds.
- TensorSharp base commit: `7a62bc10d37ec4eed4d3394af6be6b7ba11577a5` plus this change.
- TensorSharp GGML checkout: `456172ec733a135778adcd32d00e576a58232e45`.
- llama.cpp: `1bc7a5af0d14b1fb72f266abbd1237b394187115`, locally rebuilt Release.
- CPU comparisons use eight threads. The `cpu` backend is pure C#;
  `ggml_cpu` is the separate native GGML backend. Native CPU comparisons use
  quantized weight repacking in both engines.
  llama.cpp also enables Apple Accelerate BLAS; TensorSharp's GGML build has BLAS
  disabled. Metal runs use the same machine and model weights.
- Measurements use the shared desktop. Unrelated background applications remain
  running; builds, tests, and other task-owned compute are kept outside timed
  runs. Engine-order checks and the fixed repeatability follow-up retain this
  variability instead of selecting only favorable samples.

The runner starts and stops the two servers sequentially, sends real HTTP
requests, and includes tokenization, inference, output normalization, JSON
serialization, and client response parsing. It preserves each sample, p95,
tokens/second, vectors, exact commands, inputs, and model hashes in `results.json`.
There is no embedding-result cache. Model weights, native graph allocations, and
managed activation buffers remain resident between requests. Each engine
receives identical text inputs. Result folders ending in `-cpu` refer to native
`ggml_cpu`; folders containing `-managed` refer to the pure C# `cpu` backend.

The final measurements use persistent HTTP connections, matching normal SDK
usage, with routine request/file logging disabled for both servers. Both receive
ten seconds of runtime prewarm and five warmups per shape. Every standard case
records at least 20 samples and runs for at least one second; its actual sample
count is recorded.
The raw commands and connection mode make these settings explicit. Earlier
new-connection and default-logging runs remain as diagnostic evidence; they
include additional HTTP and logging overhead, significant for MiniLM's smallest
inputs. Forward and reverse engine orders are recorded separately.

The retained `minilm-managed-keys` diagnostic shows why the runtime prewarm was
extended: after two seconds, short-input sample medians still fell from about
2.2 ms to 1.0–1.1 ms as .NET promoted hot methods. The final ten-second warmup
applies equally to both engines. These measurements cover steady-state service
latency; model loading and runtime compilation are outside the timed requests.

llama.cpp has **32 parallel slots**, batch/physical batch size 8192, and total
context 262144 (8192 per slot), allowing it to batch every input in the largest
test. A one-slot exploratory run overstated TensorSharp's batch advantage and
is excluded from final comparisons. The performance gate is a TensorSharp median
no more than 5% above llama.cpp in **every** case; results are specific to these
models, workloads, builds, and hardware.

## Performance

### Native CPU and Metal: primary run

Cells show **TensorSharp / llama.cpp median milliseconds**. Lower is better.
All 28 native cases meet the predefined median gate in this run.

#### Snowflake Arctic Embed L v2.0 Q8_0

| Workload | Native CPU (`ggml_cpu`) | Metal (`ggml_metal`) |
|---|---:|---:|
| Single, short | 9.120 / 10.371 | 9.336 / 10.363 |
| Single, medium | 69.145 / 73.460 | 13.980 / 14.812 |
| Single, long | 218.653 / 224.530 | 29.167 / 30.987 |
| 8 inputs, short | 58.847 / 67.413 | 14.444 / 16.780 |
| 8 inputs, medium | 539.607 / 592.471 | 73.554 / 79.749 |
| 32 inputs, short | 232.227 / 261.477 | 33.739 / 39.698 |
| 8 inputs, mixed lengths | 488.616 / 485.728 | 63.448 / 64.848 |

Raw samples: [native CPU](snowflake-cpu/results.json), [Metal](snowflake-metal/results.json).

#### all-MiniLM-L6-v2 Q8_0

| Workload | Native CPU (`ggml_cpu`) | Metal (`ggml_metal`) |
|---|---:|---:|
| Single, short | 0.860 / 1.022 | 1.321 / 1.723 |
| Single, medium | 4.099 / 4.459 | 1.632 / 2.187 |
| Single, long | 12.510 / 13.307 | 2.445 / 3.218 |
| 8 inputs, short | 3.606 / 4.783 | 2.462 / 4.049 |
| 8 inputs, medium | 26.982 / 31.810 | 5.090 / 8.085 |
| 32 inputs, short | 12.815 / 16.369 | 6.719 / 9.747 |
| 8 inputs, mixed lengths | 26.193 / 26.879 | 4.855 / 7.175 |

Raw samples: [native CPU](minilm-cpu/results.json), [Metal](minilm-metal/results.json).

The raw files include p95 as well as medians. Earlier native CPU runs with
two-second runtime prewarm showed roughly 10 ms stalls on short requests. Those
stalls are absent in the current-build, ten-second-prewarm primary run; the
largest native CPU p95 ratio in this run is 1.009. Earlier samples remain
available as diagnostic evidence. No scheduler change was made in response to
those measurements; the rerun does not isolate compilation from other build or
system-load differences.

### Pure C# CPU: primary run

Cells show **TensorSharp / llama.cpp median milliseconds**, using eight CPU
threads in both engines. All 14 managed cases meet the same median gate.

| Workload | Snowflake | MiniLM |
|---|---:|---:|
| Single, short | 8.736 / 10.006 | 0.859 / 0.919 |
| Single, medium | 65.203 / 71.765 | 3.898 / 4.234 |
| Single, long | 222.006 / 219.107 | 12.951 / 12.878 |
| 8 inputs, short | 59.251 / 65.766 | 3.546 / 4.510 |
| 8 inputs, medium | 527.548 / 572.223 | 27.620 / 30.430 |
| 32 inputs, short | 224.814 / 254.263 | 12.484 / 15.723 |
| 8 inputs, mixed lengths | 448.666 / 472.246 | 24.081 / 25.909 |

Raw samples: [Snowflake](snowflake-managed/results.json),
[MiniLM](minilm-managed/results.json).

Some managed MiniLM cases have higher p95 despite comparable or faster medians;
the raw samples retain those outliers. Median parity does not imply parity in
every latency percentile.

### Engine-order check

Both orders are complete. The pure C# backend passes **28/28** case/order
comparisons; Metal passes **28/28**; native GGML CPU passes **27/28**. The one
miss is MiniLM's native CPU reverse-order short case: **0.981 / 0.921 ms**,
a ratio of **1.065**, just outside the unchanged 1.05 gate. That result remains
in [the raw report](minilm-cpu-reverse/results.json).

A predefined follow-up ran exactly six pairs on that case, alternating engine
order, with ten-second prewarm, five shape warmups, and at least three seconds
of sampling per engine. All six pass: ratios **0.794–1.044**, geometric mean
**0.873**. [All pairs and the fixed plan](native-minilm-short-repeatability.json)
are retained; they establish repeatability within observed machine variability
and do not erase the original miss.

Reverse-order samples: [Snowflake pure C#](snowflake-managed-reverse/results.json),
[MiniLM pure C#](minilm-managed-reverse/results.json),
[Snowflake native CPU](snowflake-cpu-reverse/results.json),
[MiniLM native CPU](minilm-cpu-reverse/results.json),
[Snowflake Metal](snowflake-metal-reverse/results.json),
[MiniLM Metal](minilm-metal-reverse/results.json).
The [matrix summary](performance-summary.json) lists all 84 comparisons.

### Full 8192-token context

The seven standard workloads reach 478 tokens for Snowflake's longest individual
input. A separate [full-context input](full-context-inputs.json) repeats `a` to
produce exactly 8192 tokens including special tokens. Both engines receive ten
seconds of short-input runtime prewarm, two full-context warmups, and three
measured full-context requests. These are expensive-shape measurements with
three samples, separately reported from the standard 20-sample matrix.
Vectors are checked for finite unit norms and same-model cosine at least 0.999.
The repeated input measures the context-size cost; it is not a long-document
retrieval quality evaluation.

The current Metal result passes: **1,704.338 / 1,882.318 ms**, cosine
**0.999999960** ([raw result](snowflake-metal-full-context/results.json)).
The optimized pure C# result passes: **15,530.190 / 15,581.754 ms**, cosine
**0.999934605** ([raw result](snowflake-managed-full-context/results.json)).
The initial managed result was 24,062.358 / 15,666.429 ms; head-major scheduling
reduced it to 22,093.590 / 15,340.177 ms, and the first 64-by-64 tiled version
reached 19,455.568 / 15,599.382 ms. All three missed the unchanged 1.05 gate.
The accepted version uses 64-query by 128-key tiles, wider ARM SIMD register
kernels, and a fused maximum-subtraction/exponential/sum pass. Earlier failures
remain in the diagnostic archive and the `-before-tiled` / `-before-wide` folders.
Native CPU's initial 18,319.074 / 15,128.357 ms result is also retained.

[Managed stage profiling](managed-full-context-phase-profile.json) attributes
about 87% of the instrumented full-context request to attention. All three
profiled vectors exactly match the uninstrumented result. The profile is a
diagnostic, not a latency benchmark. It motivated a long-input implementation
that reuses small FP32 query/key/value tiles and accumulates online softmax,
following llama.cpp's CPU attention structure.

For native CPU, the two checkouts have identical attention, SIMD GEMM, vector,
and tile-constant source. TensorSharp's original large-input K/V views had
strides through fused QKV storage; making them contiguous reduced the diagnostic
median from 18,319 to 15,682 ms with identical output. The
[native layout diagnostic](snowflake-cpu-full-context-contiguous-diagnostic/results.json)
uses the retained llama baseline; a fresh paired run is still required.
[Native boundary checks](native-long-fixture-tests.json) also compare old/new
binaries on 12 synthetic inputs and all pooling modes. Vectors remain identical.
Their scalar references account explicitly for ggml's existing FP16 GELU lookup
table; both native and managed reference errors remain below 2e-6. Existing
real-model accuracy gates are unchanged.

### Preserved exploratory runs

[Diagnostic index](diagnostics-index.json) summarizes all 42 exploratory or
superseded runs, including failed performance gates, cold-runtime measurements,
and the initial full-context gap. Their exact inputs, samples, vectors, commands,
and logs are retained in [the archive](diagnostics.tar.gz). Every archived file's
length and SHA-256 was checked before removing its uncompressed duplicate.
The index contains both per-file and archive checksums. Extract separately with:

```bash
mkdir -p /tmp/embedding-diagnostics
tar -xzf docs/validation/embeddings-2026-09/diagnostics.tar.gz -C /tmp/embedding-diagnostics
```

## Correctness and regression coverage

The final Release regression lane passes **3,908 tests, zero failures and zero
skips**. The focused embedding/kernel/worker-pool lane passes **245 tests**;
these suites overlap and their counts must not be added.

- **HTTP vector parity:** 13 code, text, multilingual, and normalization inputs;
  minimum same-model cosine 0.999, finite unit-length vectors, identical token
  counts, four retrieval rankings, batch/single consistency, reversed order,
  duplicates, and concurrent requests. TensorSharp's batch consistency gate is
  0.9999. llama.cpp's CPU batch/single differences require an explicitly recorded
  0.999 baseline threshold; its Metal threshold remains 0.9999.
- **Independent forward pass:** [NumPy oracle](numpy-oracle/README.md) runs six
  examples through every model layer using dequantized GGUF weights and FP32
  math, independently of both native engines. Predeclared gates are cosine
  at least 0.999 and maximum component error at most 0.005. llama.cpp MiniLM CPU
  narrowly exceeds the
  component-error gate on the Unicode example; that failure is retained.
  All 72 TensorSharp vectors and 48 retrieval rankings pass across the 24
  forward/reverse oracle reports.
- **Tokenizer conformance:** [HuggingFace fixtures](huggingface-tokenization.json)
  and [Snowflake llama.cpp fixtures](snowflake-tokenization.json) verify special
  tokens, the GGUF character map, unigram segmentation, WordPiece, and Unicode.
  Intentional upstream/GGUF differences are explained in the usage guide.
- **Context boundaries:** [recorded HTTP results](context-boundaries.json) verify
  Snowflake's full 8192-token context and MiniLM's 512-token context. One token
  over the limit is rejected. Ollama truncation preserves the terminal token;
  disabling truncation returns an input error on overflow.
  [Pure C# native-free full-context results](managed-context-boundaries.json)
  also pass at both limits; truncated output exactly equals explicit token
  input with its terminal token preserved, and subsequent short requests succeed.
- **Official clients:** [SDK smoke test](sdk-smoke.json) uses OpenAI Python 2.48.0
  with its default base64 decoding and Ollama Python 0.6.2, both modern and
  legacy endpoints, 256 dimensions, and consistent token accounting.
- **Native-free deployment:** [isolated host verification](managed-native-free.json)
  removes all 21 custom native assets from a copied Release host. Both models
  pass OpenAI float/base64, Ollama modern, and legacy requests on `cpu`.
  `vmmap` and `lsof` show no custom native images loaded. Standard .NET runtime
  and operating-system libraries remain necessary for any managed application.
  [The smoke runner](../../../benchmarks/EmbeddingBench/native_free_smoke.py)
  reproduces the copy, protocol checks, process inspection, and cleanup.
- **Managed encoder:** [focused tests](managed-focused-tests.json) cover both
  real models against native inference, strict batch agreement, independent
  analytic F32 fixtures for all three pooling modes, and in-flight cancellation
  followed by successful reuse. Pure C# initialization bypasses native backend
  discovery, tensor allocation, and dequantization dispatch.
- **Portable math:** [46 tests with hardware intrinsics disabled](managed-portable-tests.json)
  exercise generic score/value kernels, online softmax, scalar encoder references,
  and buffer tails. Four ARM-specific Q8 cases return when their required ISA is
  disabled. This fallback check does not measure the x86 AVX2/AVX-512 paths.
- **Existing chat:** [live Qwen3.5-9B-Q8_0 Metal regression](chat-regression.json)
  verifies OpenAI text (`42`), constrained JSON (`{"answer":42}`), and Ollama
  generation (`Paris`).
- **Managed tests:** [unit-test summary](unit-tests.json) and
  [API review tests](api-review-tests.json) cover actual Kestrel protocol routes,
  standalone embedding hosting, JSON/CLI options, bad inputs, cancellation,
  normalized float/base64 output, model discovery, mobile exports, and existing
  managed runtime behavior. The later API review adds 16 MiB body-limit checks
  for both Content-Length and chunked requests, invalid paths, and tiny/large
  normalization. Counts from overlapping suites must not be added together.
- **Concurrent dispatch:** [17 deterministic scheduler cases](dispatcher-focused-tests.json)
  cover immediate first dispatch, FIFO merging of waiting requests, sequence and
  token caps, oversized requests, result and usage splitting, mixed protocols,
  cancellation isolation, error recovery, synchronous completion, parallel
  submissions, and shutdown without taking ownership of the registered model.
  These tests are included in the 245-test focused lane.

The initial baseline found an existing Q2_K batched-row timing regression.
The shared CPU helper now decodes a quantized weight row once and reuses it over
contiguous activation rows. Scalar-reference tests cover row offsets, strides,
and tails. Three new embedding symbols were also added to the iOS linker
retention list after its regression test caught the omission.

## Implementation lessons applied

- **llama.cpp:** native BERT graph and pooling semantics, exact GGUF tokenizer
  behavior, no autoregressive KV cache, flash attention, CPU optimized weight
  buffers, and Metal graph optimization before allocation.
- **vLLM** ([pooling service](https://github.com/vllm-project/vllm/blob/ba2ae9f23961ac67bc5c055da8c26fbd660989c6/vllm/entrypoints/pooling/embed/serving.py)): distinct pooling-model
  execution, per-input pooling, output normalization, and explicit protocol
  validation.
- **SGLang** ([embedding service](https://github.com/sgl-project/sglang/blob/6e755e411440bb3e42df59a9d0bc4e250615cf09/python/sglang/srt/entrypoints/openai/serving_embedding.py)): embedding input
  validation, batched scheduling, and float32 base64 representation.

TensorSharp owns the encoder implementation. It does not invoke llama.cpp for
production inference. Q/K/V weights are fused once without dequantizing them;
graphs reuse allocations; CLS discards unnecessary final feed-forward rows;
GPU mixed batches use a block-diagonal attention mask. Performance changes are
accepted only with vector and batch-isolation checks.
Native CPU batches pack projection tokens while keeping separately padded
attention sequences. CPU attention uses F32 K/V, making them contiguous for
sequences of at least 1024 tokens, and caches its work plan alongside graph
allocations. The managed long-input path reuses FP32 tiles with online softmax;
it does not introduce reduced-precision attention or a result cache.

## Reproduce tests

```bash
bash TensorSharp.GGML.Native/build-macos.sh
dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter 'Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx'

export TENSORSHARP_SNOWFLAKE_EMBED_MODEL=/path/to/snowflake-arctic-embed-l-v2.0-q8_0.gguf
export TENSORSHARP_MINILM_EMBED_MODEL=/path/to/all-MiniLM-L6-v2-Q8_0.gguf
TS_TEST_EMBEDDING_BACKEND=GGML_CPU dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -c Release --no-build --filter 'FullyQualifiedName~EmbeddingModelTests|FullyQualifiedName~EmbeddingTokenizerTests'
TS_TEST_EMBEDDING_BACKEND=METAL dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -c Release --no-build --filter 'FullyQualifiedName~EmbeddingModelTests'
```

## Scope

Validated architectures are GGUF `bert` BERT and XLM-RoBERTa encoders, with CLS
and mean pooling exercised by real models. Last-token pooling has an independent
analytic test; no downloaded last-token encoder is represented in the benchmark. CUDA can use
a CUDA-enabled GGML build but was not available for validation here. Decoder
embedding models, rerankers, sparse/multi-vector output, and dynamic multi-model
serving are outside this change.

The small retrieval fixture is a regression check, not MTEB or a general quality
ranking. The NumPy oracle measures implementation agreement for the same GGUF,
not quantization loss against the original checkpoint. Chat and embedding models
run in separate host processes. Website changes are source updates; publishing
and a public deployment depend on the repository's deployment environment.
