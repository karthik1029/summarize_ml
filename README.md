# 📝 Text/URL Summarizer

Summarize articles using Hugging Face BART models. Runs as a lightweight Gradio UI in Docker.  
Supports URL/text input, safetensors-only models, and persistent model cache.

---

## Quick Start (Docker)

### Build the image
```bash
# (Optional on Apple Silicon) build for amd64:
# docker build --platform=linux/amd64 -t text-summarize_ml:latest .

docker build -t text-summarize_ml:latest .
```


## Persist the Hugging Face cache
```bash
docker volume create hf-cache
```


## Run the UI (BART-CNN by default)
```bash
docker run --rm -p 7860:7860 \
  -e SUMM_MODEL=facebook/bart-large-cnn \
  -e PRELOAD_AT_START=1 \
  -v hf-cache:/root/.cache/huggingface \
  --name textsum text-summarize_ml:latest
```

## Access the UI
**http://localhost:7860**

---

## Switch Models (safetensors-friendly)


### Newsy / concise (XSum style)
```bash
docker run --rm -p 7860:7860 \
  -e SUMM_MODEL=facebook/bart-large-xsum \
  -e PRELOAD_AT_START=1 \
  -v hf-cache:/root/.cache/huggingface \
  text-summarize_ml:latest
```

### Dialog / conversation (SAMSum)
```bash
docker run --rm -p 7860:7860 \
  -e SUMM_MODEL=philschmid/bart-large-cnn-samsum \
  -e PRELOAD_AT_START=1 \
  -v hf-cache:/root/.cache/huggingface \
  text-summarize_ml:latest
```



## Option: Preload **both** models into the shared cache

This warms the Hugging Face cache so switching models in the UI is instant.

```bash
# make sure the cache volume exists
docker volume create hf-cache

# preload both models (BART-CNN and BART-XSUM) into the volume
docker run --rm -v hf-cache:/root/.cache/huggingface text-summarize_ml:latest \
  python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as M; \
models=['facebook/bart-large-cnn','facebook/bart-large-xsum']; \
[ (print('Preloading',m), AutoTokenizer.from_pretrained(m), M.from_pretrained(m, use_safetensors=True)) for m in models ]"
```


**Want to include the SAMSum dialog model too? Use:**
```bash
docker run --rm -v hf-cache:/root/.cache/huggingface text-summarize_ml:latest \
  python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as M; \
models=['facebook/bart-large-cnn','facebook/bart-large-xsum','philschmid/bart-large-cnn-samsum']; \
[ (print('Preloading',m), AutoTokenizer.from_pretrained(m), M.from_pretrained(m, use_safetensors=True)) for m in models ]"
```



## Environment Variables
- **`SUMM_MODEL`** — HF model ID (default: `facebook/bart-large-cnn`)
- **`PRELOAD_AT_START`** — `1` to download/warmup the model at container start (recommended)
- **`PORT`** — internal server port (default: `7860`)

> Models should provide **`.safetensors`** weights for best compatibility.



## Notes
- The named volume `hf-cache` keeps model weights between runs (faster startups).
- If port `7860` is busy, change `-p 7860:7860` to another host port (e.g., `-p 8080:7860`).
