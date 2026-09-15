import json,sys,time,types
from pathlib import Path
sys.path.insert(0,str(Path('benchmarks/EmbeddingBench').resolve()))
import embedding_bench as bench
baseline_path=Path('docs/validation/embeddings-2026-09/snowflake-cpu-full-context/results.json')
baseline=json.loads(baseline_path.read_text())
output=Path('docs/validation/embeddings-2026-09/snowflake-cpu-full-context-contiguous-diagnostic')
output.mkdir(exist_ok=True)
name='single_full_context_8192'
bench.KEEP_ALIVE=True
bench.EXTRA_SCENARIOS={name:baseline['benchmark_inputs'][name]}
args=types.SimpleNamespace(model='snowflake-arctic-embed-l-v2.0-q8_0',reference_min_batch_cosine=.9999,prewarm_seconds=10,warmup=2,repeats=3,minimum_measure_seconds=0,cases=[name])
command=baseline['engines']['tensorsharp']['command']
artifacts=bench.command_artifacts(command)
result=bench.run_engine('tensorsharp',command,baseline['engines']['tensorsharp']['url'],args,output)
actual=result['benchmarks'][name]
previous=baseline['engines']['tensorsharp']['benchmarks'][name]
llama=baseline['engines']['llama']['benchmarks'][name]
a,b=actual['vectors'][0],previous['vectors'][0]
report=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),diagnostic=True,candidate='CPU length>=1024 packs F32 K/V to contiguous head-major tensors; all short paths unchanged',build_command='cmake --build TensorSharp.GGML.Native/build --target GgmlOps -j 8',driver_command='python3 /tmp/ts-native-long-diagnostic.py',reference_path=str(baseline_path),fresh_llama_measured=False,warmup=2,repeats=3,prewarm_seconds=10,binary_artifacts=artifacts,engine=result,comparison=dict(previous_tensorsharp_median_ms=previous['median_ms'],retained_llama_median_ms=llama['median_ms'],ratio_to_previous=actual['median_ms']/previous['median_ms'],ratio_to_retained_llama=actual['median_ms']/llama['median_ms'],cosine_to_previous=bench.cosine(a,b),maximum_absolute_error_to_previous=max(abs(x-y) for x,y in zip(a,b)),cosine_to_retained_llama=bench.cosine(a,llama['vectors'][0])))
(output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['comparison'],indent=2),flush=True)
