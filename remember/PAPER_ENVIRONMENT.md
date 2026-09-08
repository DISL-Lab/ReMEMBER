# Paper reproduction environment

This is a snapshot of the exact environment used to verify the release
package against the paper's archived outputs.

- **Retrieval/packing: 100% of 1079 windows byte-identical**, and this holds
  unconditionally — it's pure embedding/CPU computation, not affected by
  anything below. This is the part of the release that was actually broken
  (8 bugs, all fixed and verified; see commit history) and it now
  reproducibles the paper's method exactly.
- **Gap detection (LLM generation): not reliably reproducible outside this
  exact server's compiled-kernel cache** — see below. Retrieval/packing was
  fed the paper's own cached gap predictions for that 100% figure; this
  section is about regenerating gap text from scratch with a local vLLM.

It is captured here so a from-scratch setup can at least rule out **package
version drift** as a source of divergence in the parts that can be matched
(retrieval/packing) and to document, honestly, why the LLM-generated part
(gap text, and summaries) should not be expected to match byte-for-byte
anywhere but here.

## Why gap-detection text isn't reproducible, in three steps

1. **Busy server, reused compiled cache → 86%.** First attempt shared the
   server with other jobs saturating the *other* GPUs (one at 99% util, one
   holding 89GB); this run's own GPU showed idle in `nvidia-smi` beforehand,
   but 86% of 1079 windows still matched the paper's archived gap text
   exactly (batched identically: same GPU index, same 96-window batches).
2. **Quiet server, same reused compiled cache → 100%.** Re-running the
   identical batch with the whole server otherwise idle reproduced all 1079
   windows exactly. This looked like proof that "quiet GPU" was the whole
   story.
3. **Quiet server, cache cleared (forced fresh `torch.compile`) → 19.6%.**
   Moving aside `~/.cache/vllm/torch_compile_cache/<hash>/` (the compiled
   Triton/CUDA kernels vLLM's JIT compiler had produced for this exact
   model+config back when the paper's runs were made) and re-running forced
   a fresh compile on the *same* GPU, same driver, same vLLM build. Result:
   211/1079 (19.6%) exact matches. The cache — not GPU business — was doing
   nearly all the work in steps 1–2.

**Why re-compiling changes the actual output, not just speed**: `torch.compile`
picks each operator's kernel by live-benchmarking multiple candidate
implementations and keeping whichever measures fastest at that moment.
Mathematically-equivalent kernels can still sum floating-point values in a
different order, so a freshly-chosen kernel can produce a hairline-different
result for that operator. This happens independently across dozens of fused
operations in every transformer layer; gap detection is autoregressive, so
the first token where any one of those hairline differences flips an argmax
choice changes the token fed into every following step, and the completion
diverges from there. Across 1079 windows, each generating many tokens, it
would be surprising if *none* of them hit this — which is exactly what steps
1–3 show happening at scale once the specific compiled kernels are gone.

## This is a known, documented class of problem — not unique to this release

vLLM's own docs describe exactly this mechanism and only promise
reproducibility under narrower conditions than "same hardware, same vLLM
version, quiet GPU":

- [vLLM: Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/) —
  "only provides reproducibility when it runs on the same hardware and the
  same vLLM version," and even then only with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (offline, deterministic scheduling) or `VLLM_BATCH_INVARIANT=1` explicitly set.
  Neither was set for the paper's runs.
- [vLLM: Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) —
  states the root cause as "batch-size dependence of reduction kernels: a
  given prompt can be dispatched under different dynamic batch sizes,
  causing internal reduction trees to change."
- [vLLM issue #19491](https://github.com/vllm-project/vllm/issues/19491) — a
  filed bug report of exactly this symptom (AsyncLLM outputs differing
  across runs of the same script).
- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/2.13/notes/randomness.html) —
  "Completely reproducible results are not guaranteed across PyTorch
  releases, individual commits, or different platforms," true even with
  identical seeds.

## Practical takeaway

- Trust retrieval/packing to reproduce the paper exactly, anywhere, given the
  same predicted gaps as input (already verified: 1079/1079).
- Do not expect a from-scratch `--provider vllm` run's gap-detection or
  summary text to match this repository's archived numbers byte-for-byte,
  on this server or any other — treat close-but-different generations as
  expected, not as a sign something is broken. What should carry over is the
  *behavior* (similar gap types found, similar summary quality), not the
  literal text.

**Practical takeaway**: for a from-scratch reproduction attempt to match
these numbers, run it on a GPU/server with no other concurrent jobs, not
just a GPU that individually shows 0% utilization right before you start.

## Hardware / driver (as observed on the verification server)

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- NVIDIA driver: 580.126.09
- CUDA (via torch): 12.9
- Python: 3.12.13

## Core method dependencies (from `pip freeze`)

```
vllm==0.19.1rc1.dev336+g98700c610
torch==2.11.0+cu129
transformers==5.5.0
sentence-transformers==5.4.1
numpy==2.2.6
tiktoken==0.12.0
rank-bm25==0.2.2
PyYAML==6.0.3
```

## Full `pip freeze` (py312gemma conda env)

A handful of lines below (`asttokens`, `comm`, `debugpy`, `decorator`,
`executing`, `ipykernel`, `ipython`, `jedi`, `matplotlib-inline`,
`nest_asyncio`, `parso`, `pexpect`, `prompt_toolkit`, `psutil`, `ptyprocess`,
`pure_eval`, `Pygments`, `python-dateutil`, `pyzmq`, `six`, `stack_data`,
`tornado`, `traitlets`, `typing_extensions`, `wcwidth`) point at local
conda-forge build artifacts (`@ file:///home/conda/...`) rather than a
pip-installable version — they are Jupyter/IPython dev tooling unrelated to
the method itself and can be ignored (or installed from PyPI at any recent
version) if reconstructing this environment with plain pip.

```
accelerate==1.13.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anthropic==0.95.0
anyio==4.13.0
apache-tvm-ffi==0.1.10
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
astor==0.8.1
asttokens @ file:///home/conda/feedstock_root/build_artifacts/asttokens_1763409923949/work
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.14.3
bitsandbytes==0.49.2
blake3==1.0.8
bleach==6.3.0
cachetools==7.0.5
cbor2==6.0.0rc1
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.7
click==8.3.2
cloudpickle==3.1.2
comm @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_comm_1753453984/work
compressed-tensors==0.15.0.1
contourpy==1.3.3
cryptography==46.0.7
cuda-bindings==12.9.6
cuda-pathfinder==1.5.3
cuda-python==12.9.6
cuda-tile==1.2.0
cuda-toolkit==12.9.1
cycler==0.12.1
debugpy @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_debugpy_1769744978/work
decorator @ file:///home/conda/feedstock_root/build_artifacts/decorator_1740384970518/work
defusedxml==0.7.1
depyf==0.20.0
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
executing @ file:///home/conda/feedstock_root/build_artifacts/executing_1756729339227/work
fastapi==0.135.3
fastapi-cli==0.0.24
fastapi-cloud-cli==0.17.0
fastar==0.11.0
fastjsonschema==2.21.2
fastsafetensors==0.2.2
filelock==3.28.0
flashinfer-cubin==0.6.7
flashinfer-python==0.6.7
flatbuffers==25.12.19
fonttools==4.62.1
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2026.3.0
gguf==0.18.0
googleapis-common-protos==1.74.0
grpcio==1.80.0
h11==0.16.0
hf-xet==1.5.0.dev0
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
httpx-sse==0.4.3
huggingface_hub==1.10.2
idna==3.11
ijson==3.5.0
importlib_metadata==8.7.1
iniconfig==2.3.0
interegular==0.3.3
ipykernel @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_ipykernel_1770566133/work
ipython @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_ipython_1777038396/work
ipython_pygments_lexers @ file:///home/conda/feedstock_root/build_artifacts/ipython_pygments_lexers_1737123620466/work
ipywidgets==8.1.8
isoduration==20.11.0
jedi @ file:///home/conda/feedstock_root/build_artifacts/jedi_1733300866624/work
Jinja2==3.1.6
jiter==0.14.0
jmespath==1.1.0
joblib==1.5.3
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_client @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_jupyter_client_1767954655/work
jupyter_core @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_jupyter_core_1760643864/work
jupyter_server==2.18.2
jupyter_server_terminals==0.5.4
jupyterlab==4.5.7
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_widgets==3.0.16
kiwisolver==1.5.0
lark==1.2.2
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline @ file:///home/conda/feedstock_root/build_artifacts/matplotlib-inline_1761214490209/work
mcp==1.27.0
mdurl==0.1.2
mistral_common==1.11.0
mistune==3.2.1
model-hosting-container-standards==0.1.14
mpmath==1.3.0
msgspec==0.21.1
multidict==6.7.1
nbclient==0.10.4
nbconvert==7.17.1
nbformat==5.10.4
nest_asyncio @ file:///home/conda/feedstock_root/build_artifacts/nest-asyncio_1733325553580/work
networkx==3.6.1
ninja==1.13.0
nltk==3.9.4
notebook==7.5.6
notebook_shim==0.2.4
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.9.1.4
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.17.1.4
nvidia-cudnn-frontend==1.18.0
nvidia-cufft-cu12==11.4.1.4
nvidia-cufile-cu12==1.14.1.1
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.5.0.dev0
nvidia-cutlass-dsl-libs-base==4.5.0.dev0
nvidia-ml-py==13.580.82
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.9.79
nvitop==1.6.2
onnxruntime==1.27.0
openai==2.32.0
openai-harmony==0.0.8
opencv-python-headless==4.13.0.92
opentelemetry-api==1.41.0
opentelemetry-exporter-otlp==1.41.0
opentelemetry-exporter-otlp-proto-common==1.41.0
opentelemetry-exporter-otlp-proto-grpc==1.41.0
opentelemetry-exporter-otlp-proto-http==1.41.0
opentelemetry-proto==1.41.0
opentelemetry-sdk==1.41.0
opentelemetry-semantic-conventions==0.62b0
opentelemetry-semantic-conventions-ai==0.5.1
outlines_core==0.2.11
packaging==26.1
pandas==3.0.2
pandocfilters==1.5.1
parso @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_parso_1770676243/work
partial-json-parser==0.2.1.1.post7
peft==0.19.1
pexpect @ file:///home/conda/feedstock_root/build_artifacts/pexpect_1733301927746/work
pillow==12.2.0
platformdirs @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_platformdirs_1775741140/work
pluggy==1.6.0
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.25.0
prompt_toolkit @ file:///home/conda/feedstock_root/build_artifacts/prompt-toolkit_1756321756983/work
propcache==0.4.1
protobuf==6.33.6
psutil @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_psutil_1769678155/work
ptyprocess @ file:///home/conda/feedstock_root/build_artifacts/ptyprocess_1733302279685/work/dist/ptyprocess-0.7.0-py2.py3-none-any.whl#sha256=92c32ff62b5fd8cf325bec5ab90d7be3d2a8ca8c8a3813ff487a8d2002630d1f
pure_eval @ file:///home/conda/feedstock_root/build_artifacts/pure_eval_1733569405015/work
py-cpuinfo==9.0.0
pybase64==1.4.3
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.1
pydantic-extra-types==2.11.1
pydantic-settings==2.13.1
pydantic_core==2.46.1
Pygments @ file:///home/conda/feedstock_root/build_artifacts/pygments_1774796700044/work
PyJWT==2.12.1
pymupdf==1.28.0
pymupdf-layout==1.28.0
pymupdf4llm==1.28.0
pyparsing==3.3.2
pytest==9.1.1
python-dateutil @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_python-dateutil_1751104122/work
python-dotenv==1.2.2
python-json-logger==4.1.0
python-multipart==0.0.26
PyYAML==6.0.3
pyzmq @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_pyzmq_1771716961/work
quack-kernels==0.3.10
rank-bm25==0.2.2
referencing==0.37.0
regex==2026.4.4
requests==2.33.1
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rich-toolkit==0.19.7
rignore==0.7.6
rpds-py==0.30.0
safetensors==0.8.0rc0
scikit-learn==1.8.0
scipy==1.17.1
Send2Trash==2.1.0
sentence-transformers==5.4.1
sentencepiece==0.2.1
sentry-sdk==3.0.0a7
setproctitle==1.3.7
setuptools==80.10.2
shellingham==1.5.4
six @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_six_1753199211/work
sniffio==1.3.1
soupsieve==2.8.3
sse-starlette==3.3.4
stack_data @ file:///home/conda/feedstock_root/build_artifacts/stack_data_1733569443808/work
starlette==0.52.1
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
terminado==0.18.1
threadpoolctl==3.6.0
tiktoken==0.12.0
tinycss2==1.4.0
tokenizers==0.22.2
torch==2.11.0+cu129
torch_c_dlpack_ext==0.1.5
torchaudio==2.11.0+cu129
torchvision==0.26.0+cu129
tornado @ file:///home/conda/feedstock_root/build_artifacts/tornado_1774357896001/work
tqdm==4.67.3
traitlets @ file:///home/conda/feedstock_root/build_artifacts/traitlets_1733367359838/work
transformers==5.5.0
triton==3.6.0
typer==0.24.1
typing-inspection==0.4.2
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_typing_extensions_1756220668/work
tzdata==2026.2
uri-template==1.3.0
urllib3==2.6.3
uv==0.11.7
uvicorn==0.44.0
uvloop==0.22.1
vllm==0.19.1rc1.dev336+g98700c610
watchfiles==1.1.1
wcwidth @ file:///home/conda/feedstock_root/build_artifacts/wcwidth_1770634525285/work
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
websockets==16.0
wheel==0.46.3
widgetsnbextension==4.0.15
xgrammar==0.1.33
yarl==1.23.0
zipp==3.23.1
```
