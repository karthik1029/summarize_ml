#!/usr/bin/env python3
"""
Text Summarizer core module.
- Uses Hugging Face transformers pipeline
- Handles long texts by tokenizer-aware chunking
- Forces safetensors (works on Torch < 2.6)
- Dynamically adjusts max/min summary lengths to input size
"""
from __future__ import annotations
import os
from typing import List, Optional
from dataclasses import dataclass

from transformers import pipeline, AutoTokenizer

# Default to a model that provides safetensors
DEFAULT_MODEL = os.getenv("SUMM_MODEL", "facebook/bart-large-cnn")

@dataclass
class SummarizeConfig:
    model_name: str = DEFAULT_MODEL
    max_summary_tokens: int = 160
    min_summary_tokens: int = 40
    chunk_overlap: int = 50
    device: Optional[int] = None  # set to 0 for GPU if available

class TextSummarizer:
    def __init__(self, config: SummarizeConfig = SummarizeConfig()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        # Force safetensors to avoid .bin loading (Torch >=2.6 requirement)
        self.summarizer = pipeline(
            "summarization",
            model=config.model_name,
            tokenizer=self.tokenizer,
            device=config.device if config.device is not None else -1,
            model_kwargs={"use_safetensors": True},
        )
        # Safe max input length for encoder
        self.max_input_len = min(
            getattr(self.summarizer.model.config, "max_position_embeddings", 1024),
            1024
        )

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _chunk_ids(self, ids: List[int]) -> List[List[int]]:
        window = self.max_input_len - 10
        if len(ids) <= window:
            return [ids]
        chunks = []
        start = 0
        overlap = min(self.config.chunk_overlap, window // 5)
        while start < len(ids):
            end = min(start + window, len(ids))
            chunks.append(ids[start:end])
            if end == len(ids):
                break
            start = end - overlap
        return chunks

    def _dyn_lengths(self, token_count: int) -> tuple[int, int]:
        """
        Compute dynamic (max_len, min_len) based on input token count.
        - Cap max_len to input_len - 2 (minimum 8)
        - min_len <= max_len - 1
        """
        max_len = min(self.config.max_summary_tokens, max(8, token_count - 2))
        min_len = min(self.config.min_summary_tokens, max_len - 1) if max_len > 1 else 1
        return max_len, min_len

    def summarize(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        ids = self._tokenize(text)
        chunks = self._chunk_ids(ids)

        partials = []
        for chunk in chunks:
            chunk_text = self._detokenize(chunk)
            max_len, min_len = self._dyn_lengths(len(chunk))
            out = self.summarizer(
                chunk_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )[0]["summary_text"]
            partials.append(out)

        if len(partials) > 1:
            combined = " ".join(partials)
            comb_ids = self._tokenize(combined)
            max_len, min_len = self._dyn_lengths(len(comb_ids))
            out = self.summarizer(
                combined,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )[0]["summary_text"]
            return out
        return partials[0]
