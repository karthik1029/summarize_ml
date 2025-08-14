#!/usr/bin/env python3
import re
import requests
import gradio as gr
from bs4 import BeautifulSoup
from summarizer import TextSummarizer, SummarizeConfig

# ---- Model options (safetensors-friendly) ----
MODELS = [
    "facebook/bart-large-cnn",           # default (safetensors)
    "facebook/bart-large-xsum",          # concise, newsy summaries
    "philschmid/bart-large-cnn-samsum",  # dialog/conversation
]
SAFE_DEFAULT = "facebook/bart-large-cnn"

# ---- Content extraction helpers ----
READABLE_SELECTORS = [
    "article", "main", "[role=main]", "#content", ".content", ".content-body",
    ".article", ".article-body", ".article-content", ".knowledge-article",
    ".knowledgeArticle", ".slds-rich-text-editor__output", ".slds-rich-text-area",
    ".slds-rich-text-editor__textarea", ".post-content", ".entry-content",
    ".prose", ".c-article__content", "#article-body", "#main-content",
]

def extract_with_trafilatura(html: str) -> str:
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html,
            include_tables=True,
            favor_recall=True,
            with_metadata=False,
            output="txt",
        )
        if extracted:
            # collapse spaces but keep newlines somewhat
            return re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", extracted)).strip()
    except Exception:
        pass
    return ""

def extract_with_bs(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    for sel in READABLE_SELECTORS:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            text = node.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text)
    # fallback: whole page text
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)

def extract_main_text(html: str) -> str:
    t = extract_with_trafilatura(html)
    if t and len(t.split()) > 60:   # avoid footer/legal-only snippets
        return t
    return extract_with_bs(html)

def fetch_if_url(text_or_url: str) -> str:
    s = (text_or_url or "").strip()
    if re.match(r"^https?://", s, flags=re.I):
        resp = requests.get(
            s,
            timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Safari/537.36"},
        )
        resp.raise_for_status()
        return extract_main_text(resp.text)
    return s

# ---- Model selection / fallback ----
def resolve_model(name: str) -> str:
    # direct passthrough (kept for future label->id mapping if needed)
    return name

def do_summarize(text_or_url, model, max_tokens, min_tokens):
    try:
        article_text = fetch_if_url(text_or_url)
        if len(article_text.split()) < 50:
            return (
                "No usable article text extracted (page may be JS-rendered, logged-in, or paywalled). "
                "Copy the article text and paste it here or use the CLI with --file.",
                ""
            )

        chosen = resolve_model(model)
        cfg = SummarizeConfig(
            model_name=chosen,
            max_summary_tokens=int(max_tokens),
            min_summary_tokens=int(min(min_tokens, max_tokens - 1)),
        )
        try:
            ts = TextSummarizer(cfg)  # summarizer enforces safetensors in its pipeline
        except Exception as e:
            msg = str(e)
            # Auto-fallback when selected model lacks safetensors
            if "does not appear to have a file named model.safetensors" in msg:
                cfg.model_name = SAFE_DEFAULT
                ts = TextSummarizer(cfg)
                summary = ts.summarize(article_text)
                return (summary, f"Notice: '{chosen}' has no safetensors. Fell back to {SAFE_DEFAULT}.")
            raise

        summary = ts.summarize(article_text)
        return (summary, "")
    except Exception as e:
        return ("", f"Error: {type(e).__name__}: {e}")

# ---- Gradio UI ----
with gr.Blocks(title="Text Summarizer") as demo:
    gr.Markdown("# 📝 Text Summarizer\nPaste article text **or a URL** below and get a concise summary.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Input Text or URL",
                lines=12,
                placeholder="Paste article text *or* a direct article URL (https://...)"
            )
            model = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
            max_tokens = gr.Slider(60, 300, value=160, step=10, label="Max summary tokens")
            min_tokens = gr.Slider(20, 140, value=40, step=5, label="Min summary tokens")
            btn = gr.Button("Summarize")
        with gr.Column():
            out = gr.Textbox(label="Summary", lines=12)
            err = gr.Textbox(label="Errors / Notices", lines=4)
    btn.click(do_summarize, [text, model, max_tokens, min_tokens], [out, err])

if __name__ == "__main__":
    import os
    demo.launch(
        debug=True,
        show_error=True,
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
