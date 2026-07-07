# GEMMA-SERVE — OpenAI-Compatible API over the Transformers Audio Path

**Spec version:** 1.0
**Purpose:** Wrap the *already-working* Hugging Face Transformers path for
`google/gemma-4-12B-it` (4-bit, native audio) in an OpenAI-compatible HTTP API,
so it's a drop-in for BEATFORGE via a base-URL swap. Runs with **or without**
Docker.

## 0. Read this first — what to reuse, what NOT to do

- **REUSE** the model-load and audio-decode logic already proven in
  `gemma_audio_eval.py` (4-bit bnb nf4 load, 16 kHz mono resample, the
  `{"type":"audio","audio": <np.ndarray>}` content-block into
  `processor.apply_chat_template`). That path is the one confirmed to actually
  do Gemma 4 audio. Do not reinvent it.
- **DO NOT** substitute vLLM, TGI, llama.cpp, or any other backend. The whole
  point is to keep the Transformers path whose audio support is verified. The
  server is a thin HTTP shell *around* that exact inference code — nothing more.
- **DO NOT** add quantization schemes other than bitsandbytes (AWQ/GPTQ risk
  stripping the multimodal projector).

## 1. Goal & non-goals

**Goal:** A single FastAPI process that loads the model once and serves
`POST /v1/chat/completions` (OpenAI schema, including `input_audio` content
blocks), plus `GET /v1/models` and `GET /health`. BEATFORGE points its OpenAI
client at `http://<host>:8000/v1` and swaps the model name — nothing else changes.

**Non-goals:** multi-model hosting, multi-GPU sharding, batching/continuous
batching (that's what vLLM is for — explicitly out of scope), auth, rate limiting.
Single model, single GPU, correctness over throughput.

## 2. Architecture

One model instance resident in VRAM. Generation is **serialized** — a single GPU
model cannot safely run concurrent `generate()` calls. Requests queue; each
`generate()` runs in a threadpool (so the event loop isn't blocked) but is guarded
by a single-flight lock. This is a correctness requirement, not an optimization.

```
client ──HTTP──▶ FastAPI (uvicorn)
                   ├─ GET /health          → {"status":"ok","model":...,"ready":bool}
                   ├─ GET /v1/models        → OpenAI model list ({served_name})
                   └─ POST /v1/chat/completions
                         ├─ translate OpenAI messages → Gemma chat messages
                         │     (decode input_audio base64 → 16 kHz mono np array)
                         ├─ acquire generation lock (serialize)
                         ├─ run in threadpool: apply_chat_template → generate → decode
                         └─ format OpenAI response (or SSE stream)
```

## 3. Configuration (all env, with defaults)

| env | default | meaning |
|---|---|---|
| `GEMMA_MODEL` | `google/gemma-4-12B-it` | HF repo id |
| `SERVED_MODEL_NAME` | `gemma-4-12b` | name clients pass as `model` |
| `HF_TOKEN` | — | required (gated model) |
| `HF_HOME` | `/models` (Docker) / `~/.cache/huggingface` (bare) | weight cache |
| `QUANT` | `4bit` | `4bit` \| `8bit` \| `bf16` |
| `MAX_AUDIO_SECONDS` | `30` | clip cap (Gemma audio ceiling); longer is truncated + warned |
| `DEFAULT_MAX_TOKENS` | `1024` | used when request omits `max_tokens` |
| `HOST` | `0.0.0.0` | bind host |
| `PORT` | `8000` | bind port |

`4bit` → bnb nf4 (~7 GB VRAM). `8bit` → bnb int8 (~12 GB). `bf16` → unquantized
(~24 GB; won't fit a 24 GB card with headroom — for A100/H100 only).

## 4. Model loading (reference — adapt from the eval script)

Load once in FastAPI's startup/lifespan; set a `ready` flag when done so `/health`
reports readiness during the download.

```python
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as AutoMM
except Exception:
    from transformers import AutoModelForCausalLM as AutoMM
import torch, os

quant = os.environ.get("QUANT", "4bit")
kw = dict(token=os.environ.get("HF_TOKEN"), device_map="auto", torch_dtype=torch.bfloat16)
if quant == "4bit":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
elif quant == "8bit":
    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL, token=kw["token"])
model = AutoMM.from_pretrained(MODEL, **kw).eval()
```

**REQ-LOAD-01.** Model loads once at startup with the configured quant; `/health`
returns `ready:false` until load completes, then `ready:true`.
`Accept:` hitting `/health` during download returns 200 with `ready:false`; after
load, `ready:true` and `/v1/chat/completions` succeeds.

## 5. API surface

### 5.1 `POST /v1/chat/completions`

Accepts the OpenAI request body. Fields honored: `model`, `messages`, `max_tokens`
(→ `max_new_tokens`), `temperature` (0 → greedy `do_sample=False`; >0 → sampling),
`top_p`, `stream`. Unknown fields are ignored, not errored.

**Message translation (the core work).** For each message, map `content`:
- string → one text block.
- list of parts:
  - `{"type":"text","text":...}` → `{"type":"text","text":...}`
  - `{"type":"input_audio","input_audio":{"data":<b64>,"format":<fmt>}}` →
    decode and resample to a 16 kHz mono array, emit `{"type":"audio","audio": arr}`
  - (optional convenience) `{"type":"audio","audio":"<local path>"}` → load path

Audio decode (reference):
```python
import base64, io, librosa
raw = base64.b64decode(part["input_audio"]["data"])
arr, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
cap = int(MAX_AUDIO_SECONDS * 16000)
if len(arr) > cap: arr = arr[:cap]           # truncate + log a warning
```

Then the SAME inference call as the eval script:
```python
inputs = processor.apply_chat_template(
    gemma_messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt")
inputs = {k: (v.to(model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=mnt, do_sample=(temp>0),
                         temperature=temp or None, top_p=top_p or None)
text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
```

**Response body** (non-streaming) — OpenAI `chat.completion`:
```json
{"id":"chatcmpl-<uuid>","object":"chat.completion","created":<epoch>,
 "model":"<served_name>",
 "choices":[{"index":0,"message":{"role":"assistant","content":"<text>"},
             "finish_reason":"stop"}],
 "usage":{"prompt_tokens":<in_len>,"completion_tokens":<out_len>,
          "total_tokens":<sum>}}
```
`prompt_tokens` = `inputs["input_ids"].shape[-1]` (audio tokens already included);
`completion_tokens` = generated token count.

**REQ-API-01.** A request with a text-only message returns a valid OpenAI
`chat.completion` object.
`Accept:` an `openai` Python client pointed at the endpoint gets a normal
completion with populated `usage`.

**REQ-API-02.** A request with an `input_audio` block returns a read that
demonstrably reflects the audio (not a generic "no audio" reply).
`Accept:` sending a 30 s clip with a "describe the structure" prompt yields
section/timestamp content that changes when the clip changes.

**REQ-API-03.** `temperature:0` is deterministic (greedy); repeated identical
requests return identical text.
`Accept:` two identical `temperature:0` calls produce byte-identical `content`.

### 5.2 `GET /v1/models`
Return `{"object":"list","data":[{"id":"<served_name>","object":"model",
"created":<epoch>,"owned_by":"local"}]}`. OpenAI clients probe this.

### 5.3 `GET /health`
`{"status":"ok","model":"<served_name>","ready":<bool>}`.

### 5.4 Errors
On failure return OpenAI-style envelope with the right HTTP status:
`{"error":{"message":<str>,"type":<str>,"code":<str|null>}}`.

## 6. Concurrency & generation

**REQ-CONC-01.** Generation is serialized by a single asyncio lock; the blocking
`generate()` runs via `run_in_executor` so the event loop stays responsive.
Concurrent callers queue rather than racing the GPU.
`Accept:` two simultaneous requests both return correct results; logs show they
generated sequentially, and VRAM does not spike from parallel generates.

## 7. Streaming (SHOULD)

Support `"stream": true` via SSE. Use `transformers.TextIteratorStreamer`: launch
`model.generate(streamer=streamer, ...)` on a background thread (still inside the
serialization lock), and yield OpenAI `chat.completion.chunk` events with
`choices[0].delta.content`, terminating with `data: [DONE]`.

**REQ-STREAM-01.** With `stream:true`, the endpoint emits incremental
`chat.completion.chunk` SSE frames ending in `[DONE]`; with `stream:false`, one
full object.
`Accept:` an OpenAI client with `stream=True` receives progressive tokens; the
same prompt non-streamed yields the same final text.

## 8. Running it

### 8.1 Without Docker (bare host / WSL2 with CUDA)
Prereqs: NVIDIA GPU + driver, a CUDA-enabled PyTorch, ffmpeg + libsndfile.
```bash
pip install -r requirements.txt            # see 8.3
export HF_TOKEN=hf_xxx
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 8.2 With Docker
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ENV HF_HOME=/models HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
```
```powershell
docker run --rm --gpus all -e HF_TOKEN=$env:HF_TOKEN `
  -v ${PWD}\models:/models -p 8000:8000 gemma-serve
```
Same `app.py` in both paths — Docker only supplies the environment.

### 8.3 requirements.txt
```
transformers        # must include Gemma 4 unified support; else install from git
accelerate
bitsandbytes
librosa
soundfile
huggingface_hub
hf_transfer
fastapi
uvicorn[standard]
```
Unpinned to grab a Gemma-4-capable build; `pip freeze` and pin after first success.

## 9. BEATFORGE swap-in

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # key ignored
# model = "gemma-4-12b"
# audio → {"type":"input_audio","input_audio":{"data": b64_wav, "format":"wav"}}
```

## 10. The one thing to VERIFY (don't assume)

The exact audio content-block key the processor wants
(`{"type":"audio","audio": array}` vs a path vs `{"array","sampling_rate"}`) is the
single Gemma-4-new detail. **Isolate it in one `to_gemma_audio_block()` function**
so if `apply_chat_template` rejects the audio, it's a one-line fix in one place.
Confirm against the `google/gemma-4-12B-it` model card before load-testing.

## 11. Definition of done

REQ-LOAD/API/CONC pass; text and audio requests both work through a stock `openai`
client; `temperature:0` is deterministic; runs identically with `uvicorn` bare and
in Docker; streaming works if implemented. Audio quality is validated by sending
two different clips and confirming the structure reads differ — same bar the eval
script had to clear.
