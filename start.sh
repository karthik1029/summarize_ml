#!/usr/bin/env bash
set -euo pipefail

if [ "${PRELOAD_AT_START:-0}" = "1" ]; then
  python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
m = os.environ.get("SUMM_MODEL", "facebook/bart-large-cnn")
print("Preloading at container start:", m, flush=True)
AutoTokenizer.from_pretrained(m)
AutoModelForSeq2SeqLM.from_pretrained(m, use_safetensors=True)
print("Warmup done.", flush=True)
PY
fi

# Run Gradio app bound to 0.0.0.0 and PORT
python - <<'PY'
import os, app
app.demo.launch(
    debug=True,
    show_error=True,
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.getenv("PORT", "7860")),
)
PY
