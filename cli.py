#!/usr/bin/env python3
"""
CLI summarizer
Usage:
  python cli.py --url https://example.com/article
  python cli.py --file article.txt
  cat article.txt | python cli.py
"""
import argparse, sys, os, re, requests
from bs4 import BeautifulSoup
from summarizer import TextSummarizer, SummarizeConfig

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
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)

def extract_main_text(html: str) -> str:
    t = extract_with_trafilatura(html)
    if t and len(t.split()) > 60:
        return t
    return extract_with_bs(html)

def read_input(args) -> str:
    if args.url:
        resp = requests.get(
            args.url, timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Safari/537.36"}
        )
        resp.raise_for_status()
        return extract_main_text(resp.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --url or --file or pipe text via stdin.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str)
    ap.add_argument("--file", type=str)
    ap.add_argument("--model", type=str, default=os.getenv("SUMM_MODEL", "facebook/bart-large-cnn"))
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--min_tokens", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = SummarizeConfig(model_name=args.model, max_summary_tokens=args.max_tokens, min_summary_tokens=args.min_tokens)
    ts = TextSummarizer(cfg)

    text = read_input(args)
    if len(text.split()) < 50:
        msg = ("No usable article text extracted (page may be JS-rendered, logged-in, or paywalled). "
               "Open the page in your browser, copy the visible article text to a file, and use --file.")
        if args.json:
            import json; print(json.dumps({"error": msg}, ensure_ascii=False))
            return
        print(msg)
        return

    summary = ts.summarize(text)
    if args.json:
        import json
        print(json.dumps({"summary": summary}, ensure_ascii=False))
    else:
        print(summary)

if __name__ == "__main__":
    main()
