from __future__ import annotations

from typing import Any

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def _normalize(x: np.ndarray) -> np.ndarray:
    # Нормализует вектор или строки матрицы по L2-норме.
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _aggregate_segment(x: np.ndarray, aggregation: str) -> np.ndarray:
    # Превращает эмбеддинги одного сегмента в один итоговый вектор.
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return _normalize(x)

    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Сегмент должен быть вектором или матрицей shape=(n_sentences, dim).")

    if aggregation == "mean":
        return _normalize(x.mean(axis=0))
    if aggregation == "median":
        return _normalize(np.median(x, axis=0))

    raise ValueError("aggregation должен быть 'mean' или 'median'.")


def _flatten_segments(
    segmented_embeddings: dict[str, list[np.ndarray]],
    segmented_texts: dict[str, list[str]],
    aggregation: str,
) -> tuple[np.ndarray, list[str], list[dict[str, int | str]]]:
    # Разворачивает корпус сегментов всех занятий в плоские списки векторов, текстов и метаданных.
    if not segmented_embeddings:
        raise ValueError("segmented_embeddings пуст.")
    if set(segmented_embeddings) != set(segmented_texts):
        raise ValueError("Ключи segmented_embeddings и segmented_texts должны совпадать.")

    vectors: list[np.ndarray] = []
    texts: list[str] = []
    meta: list[dict[str, int | str]] = []

    for lesson_name, lesson_segments in segmented_embeddings.items():
        lesson_texts = segmented_texts[lesson_name]
        if len(lesson_segments) != len(lesson_texts):
            raise ValueError(
                f"Для занятия '{lesson_name}' число сегментов embeddings и texts не совпадает."
            )

        for segment_id, (segment_embs, segment_text) in enumerate(zip(lesson_segments, lesson_texts)):
            x = np.asarray(segment_embs)
            vectors.append(_aggregate_segment(x, aggregation))
            texts.append(segment_text)
            meta.append(
                {
                    "lesson_name": lesson_name,
                    "segment_id": segment_id,
                    "n_sentences": int(x.shape[0]) if x.ndim == 2 else 1,
                    "n_tokens": len(segment_text.split()),
                }
            )

    if not vectors:
        raise ValueError("Корпус сегментов пуст.")

    return np.vstack(vectors), texts, meta


def _fit_topics(
    texts: list[str],
    vectors: np.ndarray,
    *,
    min_topic_size: int,
    top_n_words: int,
    ngram_range: tuple[int, int],
    stop_words: list[str] | str | None,
    language: str,
    nr_topics: int | str | None,
    umap_n_neighbors: int,
    umap_n_components: int,
    umap_min_dist: float,
    umap_metric: str,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, BERTopic | None]:
    # Запускает BERTopic на сегментах и возвращает метки тем и саму обученную модель.
    x = np.asarray(vectors, dtype=float)

    if len(texts) != x.shape[0]:
        raise ValueError("Число текстов сегментов должно совпадать с числом векторов.")

    if x.shape[0] == 1:
        return np.array([0], dtype=int), None

    if hdbscan_min_cluster_size is None:
        hdbscan_min_cluster_size = min_topic_size

    model = BERTopic(
        language=language,
        embedding_model=None,
        umap_model=UMAP(
            n_neighbors=min(umap_n_neighbors, max(2, x.shape[0] - 1)),
            n_components=min(umap_n_components, max(2, x.shape[0] - 1)),
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        ),
        vectorizer_model=CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
        ),
        top_n_words=top_n_words,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        nr_topics=nr_topics,
        verbose=False,
    )

    labels, _ = model.fit_transform(documents=texts, embeddings=x)
    return np.asarray(labels, dtype=int), model


def _build_topic_info(topic_model: BERTopic | None, labels: np.ndarray) -> dict[int, dict[str, Any]]:
    # Собирает краткую информацию по каждой найденной теме: имя, ключевые слова и размер.
    labels = np.asarray(labels, dtype=int)

    if topic_model is None:
        return {
            0: {
                "topic_id": 0,
                "topic_name": "topic_0",
                "keywords": [],
                "size": int(len(labels)),
            }
        }

    info_df = topic_model.get_topic_info()
    result: dict[int, dict[str, Any]] = {}

    for topic_id in sorted(set(labels.tolist())):
        topic_id = int(topic_id)
        row = info_df[info_df["Topic"] == topic_id]
        topic_name = "outlier_topic" if topic_id == -1 else f"topic_{topic_id}"

        if not row.empty and "Name" in row.columns:
            topic_name = str(row.iloc[0]["Name"])

        keywords = []
        if topic_id != -1:
            keywords = [(str(word), float(score)) for word, score in (topic_model.get_topic(topic_id) or [])]

        result[topic_id] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "keywords": keywords,
            "size": int(np.sum(labels == topic_id)),
        }

    return result


def _build_lesson_profiles(
    labels: np.ndarray,
    meta: list[dict[str, int | str]],
    *,
    weight_mode: str,
    ignore_outliers: bool,
) -> dict[str, dict[int, float]]:
    # Строит профиль каждого занятия как распределение весов по темам.
    weights: dict[str, dict[int, float]] = {}

    for label, item in zip(labels, meta):
        topic_id = int(label)
        if ignore_outliers and topic_id == -1:
            continue

        lesson_name = str(item["lesson_name"])

        if weight_mode == "token_count":
            weight = float(max(1, int(item["n_tokens"])))
        elif weight_mode == "sentence_count":
            weight = float(max(1, int(item["n_sentences"])))
        elif weight_mode == "segment_count":
            weight = 1.0
        else:
            raise ValueError("weight_mode должен быть 'token_count', 'sentence_count' или 'segment_count'.")

        weights.setdefault(lesson_name, {})
        weights[lesson_name][topic_id] = weights[lesson_name].get(topic_id, 0.0) + weight

    result: dict[str, dict[int, float]] = {}
    for lesson_name, topic_weights in weights.items():
        total = sum(topic_weights.values()) + 1e-12
        result[lesson_name] = {topic_id: w / total for topic_id, w in topic_weights.items()}

    return result


def _restore_assignments(
    labels: np.ndarray,
    meta: list[dict[str, int | str]],
    segmented_texts: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    # Восстанавливает вложенную структуру: для каждого занятия список его сегментов с присвоенными темами.
    result = {lesson_name: [] for lesson_name in segmented_texts}

    for label, item in zip(labels, meta):
        lesson_name = str(item["lesson_name"])
        segment_id = int(item["segment_id"])

        result[lesson_name].append(
            {
                "segment_id": segment_id,
                "topic_id": int(label),
                "text": segmented_texts[lesson_name][segment_id],
                "n_sentences": int(item["n_sentences"]),
                "n_tokens": int(item["n_tokens"]),
            }
        )

    for lesson_name in result:
        result[lesson_name].sort(key=lambda x: x["segment_id"])

    return result


def topicize_segmented_corpus(
    segmented_embeddings: dict[str, list[np.ndarray]],
    segmented_texts: dict[str, list[str]],
    *,
    aggregation: str = "mean",
    normalize_segment_vectors: bool = True,
    min_topic_size: int = 2,
    top_n_words: int = 8,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: list[str] | str | None = None,
    language: str = "multilingual",
    calculate_probabilities: bool = False,
    nr_topics: int | str | None = None,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
    lesson_profile_weight_mode: str = "token_count",
    ignore_outliers_in_profiles: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    # Полный пайплайн тематизации: сворачивает сегменты, обучает BERTopic и возвращает темы и профили занятий.
    segment_vectors, flat_texts, meta = _flatten_segments(
        segmented_embeddings=segmented_embeddings,
        segmented_texts=segmented_texts,
        aggregation=aggregation,
    )

    if normalize_segment_vectors:
        segment_vectors = _normalize(segment_vectors)

    labels, topic_model = _fit_topics(
        texts=flat_texts,
        vectors=segment_vectors,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        ngram_range=ngram_range,
        stop_words=stop_words,
        language=language,
        nr_topics=nr_topics,
        umap_n_neighbors=umap_n_neighbors,
        umap_n_components=umap_n_components,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        random_state=random_state,
    )

    return {
        "segment_topic_assignments": _restore_assignments(labels, meta, segmented_texts),
        "lesson_topic_profiles": _build_lesson_profiles(
            labels,
            meta,
            weight_mode=lesson_profile_weight_mode,
            ignore_outliers=ignore_outliers_in_profiles,
        ),
        "topic_info": _build_topic_info(topic_model, labels),
    }
