from __future__ import annotations

import math
from typing import Any

import numpy as np
import ruptures as rpt


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    # Нормализует каждый эмбеддинг по L2-норме.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _validate_embeddings(x: np.ndarray, min_size: int) -> np.ndarray:
    # Проверяет форму входа и минимально допустимую длину текста.
    x = np.asarray(x, dtype=float)

    if x.ndim != 2:
        raise ValueError(
            "Ожидается двумерный массив формы (n_sentences, embedding_dim)."
        )

    n_sentences = x.shape[0]
    if n_sentences < max(2, 2 * min_size):
        raise ValueError(
            f"Слишком мало предложений ({n_sentences}) для min_size={min_size}."
        )

    return x


def _map_kernel_for_article(kernel: str, normalize_embeddings: bool) -> tuple[str, bool]:
    # Приводит kernel к варианту, согласованному со статьей.
    if kernel == "cosine":
        return "linear", True
    return kernel, normalize_embeddings


def _compute_penalty(c: float, n_sentences: int) -> float:
    # Считает штраф beta_T = C * sqrt(T * log(T)).
    if c <= 0:
        raise ValueError("c должен быть положительным.")
    if n_sentences < 2:
        raise ValueError("n_sentences должен быть >= 2.")
    return float(c) * math.sqrt(n_sentences * math.log(n_sentences))


def _run_kernel_cpd(
    x: np.ndarray,
    *,
    kernel: str,
    c: float,
    min_size: int,
    normalize_embeddings: bool,
    kernel_params: dict[str, Any] | None = None,
) -> list[int]:
    # Запускает KernelCPD для одного текста и возвращает границы сегментов.
    x = _validate_embeddings(x, min_size=min_size)

    kernel_used, normalize_embeddings = _map_kernel_for_article(
        kernel, normalize_embeddings
    )

    if normalize_embeddings:
        x = _normalize_rows(x)

    penalty = _compute_penalty(c, x.shape[0])

    algo = rpt.KernelCPD(
        kernel=kernel_used,
        min_size=min_size,
        params=kernel_params,
    )
    bkps = algo.fit_predict(x, pen=penalty)
    return bkps


def segment_with_embed_kcpd(
    sentence_embeddings: np.ndarray,
    *,
    c: float,
    kernel: str = "cosine",
    min_size: int = 2,
    normalize_embeddings: bool = True,
    kernel_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Сегментирует один текст по уже выбранному глобальному C.
    x = _validate_embeddings(sentence_embeddings, min_size=min_size)

    kernel_used, normalize_embeddings = _map_kernel_for_article(
        kernel, normalize_embeddings
    )

    if normalize_embeddings:
        x_used = _normalize_rows(x)
    else:
        x_used = x

    penalty = _compute_penalty(c, x_used.shape[0])

    algo = rpt.KernelCPD(
        kernel=kernel_used,
        min_size=min_size,
        params=kernel_params,
    )
    bkps = algo.fit_predict(x_used, pen=penalty)

    change_points = bkps[:-1]

    segments: list[tuple[int, int]] = []
    start = 0
    for end in bkps:
        segments.append((start, end))
        start = end

    return {
        "change_points": change_points,
        "segments": segments,
        "penalty": penalty,
        "c": float(c),
        "kernel_used": kernel_used,
        "normalized": normalize_embeddings,
    }


def _elbow_index_max_distance(xs: np.ndarray, ys: np.ndarray) -> int:
    # Находит индекс локтя по максимальному расстоянию до хорды.
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if len(xs) < 3:
        return len(xs) // 2

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if x_max == x_min:
        x_norm = np.zeros_like(xs)
    else:
        x_norm = (xs - x_min) / (x_max - x_min)

    if y_max == y_min:
        y_norm = np.zeros_like(ys)
    else:
        y_norm = (ys - y_min) / (y_max - y_min)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)

    if line_norm == 0:
        return len(xs) // 2

    distances = []
    for i in range(len(xs)):
        p = np.array([x_norm[i], y_norm[i]])
        dist = abs(np.cross(line_vec, p - p1)) / line_norm
        distances.append(dist)

    return int(np.argmax(distances))


def select_global_c(
    corpus_embeddings: list[np.ndarray],
    *,
    kernel: str = "cosine",
    c_grid: np.ndarray | None = None,
    aggregation: str = "mean",
    min_size: int = 2,
    normalize_embeddings: bool = True,
    kernel_params: dict[str, Any] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    # Подбирает один глобальный C по набору текстов через elbow method.
    if not corpus_embeddings:
        raise ValueError("corpus_embeddings пуст.")

    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation должен быть 'mean' или 'median'.")

    if c_grid is None:
        c_grid = np.logspace(-3, 1, 25)

    c_grid = np.asarray(c_grid, dtype=float)
    if c_grid.ndim != 1 or len(c_grid) < 3:
        raise ValueError("c_grid должен быть одномерным массивом длины >= 3.")
    if np.any(c_grid <= 0):
        raise ValueError("Все значения c_grid должны быть положительными.")

    kernel_used, normalize_embeddings = _map_kernel_for_article(
        kernel, normalize_embeddings
    )

    valid_docs: list[tuple[int, np.ndarray]] = []
    for idx, doc in enumerate(corpus_embeddings):
        doc_arr = np.asarray(doc, dtype=float)
        if doc_arr.ndim != 2:
            continue
        if doc_arr.shape[0] < max(2, 2 * min_size):
            continue
        valid_docs.append((idx, doc_arr))

    if not valid_docs:
        raise ValueError("В корпусе нет ни одного валидного документа для подбора C.")

    if sample_size is None or sample_size >= len(valid_docs):
        sampled = valid_docs
    else:
        rng = np.random.default_rng(random_state)
        chosen_positions = rng.choice(len(valid_docs), size=sample_size, replace=False)
        sampled = [valid_docs[pos] for pos in chosen_positions]

    doc_c_values: list[float] = []
    sampled_indices: list[int] = []
    per_doc_curves: list[dict[str, Any]] = []

    for original_idx, doc_embeddings in sampled:
        n_changes_list = []

        for c_candidate in c_grid:
            bkps = _run_kernel_cpd(
                doc_embeddings,
                kernel=kernel,
                c=float(c_candidate),
                min_size=min_size,
                normalize_embeddings=normalize_embeddings,
                kernel_params=kernel_params,
            )
            n_changes = len(bkps) - 1
            n_changes_list.append(n_changes)

        n_changes_arr = np.asarray(n_changes_list, dtype=float)
        elbow_idx = _elbow_index_max_distance(np.log10(c_grid), n_changes_arr)
        selected_c = float(c_grid[elbow_idx])

        sampled_indices.append(original_idx)
        doc_c_values.append(selected_c)
        per_doc_curves.append(
            {
                "document_index": original_idx,
                "c_grid": c_grid.tolist(),
                "num_change_points": n_changes_arr.tolist(),
                "selected_index": int(elbow_idx),
                "selected_c": selected_c,
            }
        )

    if aggregation == "mean":
        global_c = float(np.mean(doc_c_values))
    else:
        global_c = float(np.median(doc_c_values))

    return {
        "global_c": global_c,
        "doc_c_values": doc_c_values,
        "sampled_indices": sampled_indices,
        "per_doc_curves": per_doc_curves,
        "aggregation": aggregation,
        "kernel_requested": kernel,
        "kernel_used": kernel_used,
        "normalized": normalize_embeddings,
    }
