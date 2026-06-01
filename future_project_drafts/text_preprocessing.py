import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def _normalize_transcript_text(text: str) -> str:
    text = text.strip()

    replacements = {
        "—": "-",
        "–": "-",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _filter_sentences(sentences: List[str], min_sentence_len: int = 20) -> List[str]:
    filtered = []

    for sent in sentences:
        sent = re.sub(r"\s+", " ", sent).strip()

        if len(sent) < min_sentence_len:
            continue

        filtered.append(sent)

    return filtered


def preprocess_lesson_transcript(
    text: str,
    min_sentence_len: int = 20,
) -> Dict[str, object]:
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")

    if not text.strip():
        raise ValueError("Передан пустой текст")

    cleaned_text = _normalize_transcript_text(text)
    raw_sentences = _split_into_sentences(cleaned_text)
    sentences = _filter_sentences(raw_sentences, min_sentence_len=min_sentence_len)

    if not sentences:
        raise ValueError(
            "После предобработки не осталось пригодных предложений. "
            "Возможно, текст слишком короткий или фильтрация слишком строгая."
        )

    sentence_embeddings = DEFAULT_MODEL.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return {
        "cleaned_text": cleaned_text,
        "sentences": sentences,
        "sentence_embeddings": sentence_embeddings,
    }
