from __future__ import annotations

import math
import random
from typing import Any, Hashable, Sequence

import numpy as np


LessonId = Hashable


def _get_similarity(lesson_similarity: dict[str, Any]):
    # Извлекает из матрицы схожести функцию доступа вида similarity(a, b).
    lesson_names = list(lesson_similarity["lesson_names"])
    matrix = np.asarray(lesson_similarity["similarity_matrix"], dtype=float)
    idx = {name: i for i, name in enumerate(lesson_names)}

    def get(a: LessonId, b: LessonId) -> float:
        return float(matrix[idx[a], idx[b]])

    return lesson_names, get


def _normalize_lesson_edges(
    lesson_prerequisite_edges: dict[str, Any] | Sequence[tuple[LessonId, LessonId] | tuple[LessonId, LessonId, float]] | None,
) -> list[tuple[LessonId, LessonId, float]]:
    # Приводит prerequisite-связи между занятиями к единому формату (lesson_a, lesson_b, weight).
    if lesson_prerequisite_edges is None:
        return []

    edges = (
        lesson_prerequisite_edges["lesson_edges"]
        if isinstance(lesson_prerequisite_edges, dict)
        else lesson_prerequisite_edges
    )

    result = []
    for edge in edges:
        if len(edge) == 2:
            a, b = edge
            result.append((a, b, 1.0))
        else:
            a, b, w = edge
            result.append((a, b, float(w)))
    return result


def build_lesson_prerequisite_edges_from_topic_results(
    lesson_topic_profiles: dict[str, dict[int, float]],
    prerequisite_results: list[dict[str, Any]],
    *,
    topic_edge_weight_mode: str = "prs_mean",
    lesson_edge_weight_mode: str = "product",
    min_prs_mean: float = 0.0,
    min_direction_margin: float = 0.0,
    min_topic_weight_in_lesson: float = 0.0,
    drop_self_edges: bool = True,
) -> dict[str, Any]:
    # Переводит prerequisite-связи между темами в взвешенные prerequisite-связи между занятиями.
    topic_edges: list[tuple[int, int, float]] = []

    for result in prerequisite_results:
        prs_mean = float(result.get("csr", {}).get("prs_mean", 0.0))
        margin = float(result.get("direction_scores", {}).get("direction_margin", 0.0))
        direction = str(result.get("decision", {}).get("direction", "none"))

        if prs_mean < min_prs_mean or margin < min_direction_margin:
            continue

        if direction == "a->b":
            a, b = int(result["topic_a_id"]), int(result["topic_b_id"])
        elif direction == "b->a":
            a, b = int(result["topic_b_id"]), int(result["topic_a_id"])
        else:
            continue

        if topic_edge_weight_mode == "direction_margin":
            w = margin
        elif topic_edge_weight_mode == "prs_mean_times_margin":
            w = prs_mean * margin
        else:
            w = prs_mean

        topic_edges.append((a, b, float(w)))

    lesson_edge_weights: dict[tuple[str, str], float] = {}

    for topic_a, topic_b, edge_weight in topic_edges:
        for lesson_a, profile_a in lesson_topic_profiles.items():
            weight_a = float(profile_a.get(topic_a, 0.0))
            if weight_a <= min_topic_weight_in_lesson:
                continue

            for lesson_b, profile_b in lesson_topic_profiles.items():
                weight_b = float(profile_b.get(topic_b, 0.0))
                if weight_b <= min_topic_weight_in_lesson:
                    continue
                if drop_self_edges and lesson_a == lesson_b:
                    continue

                if lesson_edge_weight_mode == "min":
                    lesson_weight = edge_weight * min(weight_a, weight_b)
                elif lesson_edge_weight_mode == "sqrt":
                    lesson_weight = edge_weight * math.sqrt(weight_a * weight_b)
                else:
                    lesson_weight = edge_weight * weight_a * weight_b

                key = (lesson_a, lesson_b)
                lesson_edge_weights[key] = lesson_edge_weights.get(key, 0.0) + lesson_weight

    lesson_edges = sorted(
        [
            (lesson_a, lesson_b, float(weight))
            for (lesson_a, lesson_b), weight in lesson_edge_weights.items()
            if weight > 0.0
        ],
        key=lambda x: x[2],
        reverse=True,
    )

    return {"lesson_edges": lesson_edges, "topic_edges": topic_edges}


def evaluate_lesson_order_coherence(
    lesson_order: Sequence[LessonId],
    lesson_similarity: dict[str, Any],
    lesson_prerequisite_edges: dict[str, Any] | Sequence[tuple[LessonId, LessonId] | tuple[LessonId, LessonId, float]] | None = None,
    *,
    similarity_weight: float = 1.0,
    closeness_decay: float = 0.85,
    prerequisite_weight: float = 10.0,
    max_similarity_distance: int | None = None,
    return_components: bool = False,
) -> float | dict[str, float]:
    # Считает итоговую согласованность порядка занятий как награду за близость схожих тем минус штрафы за нарушение prerequisite-связей.
    _, get_similarity = _get_similarity(lesson_similarity)
    lesson_order = list(lesson_order)
    position = {lesson: i for i, lesson in enumerate(lesson_order)}
    prerequisite_edges = _normalize_lesson_edges(lesson_prerequisite_edges)

    similarity_reward = 0.0
    for i, lesson_a in enumerate(lesson_order):
        j_stop = len(lesson_order) if max_similarity_distance is None else min(len(lesson_order), i + 1 + max_similarity_distance)

        for j in range(i + 1, j_stop):
            sim = get_similarity(lesson_a, lesson_order[j])
            if sim > 0.0:
                similarity_reward += similarity_weight * sim * (closeness_decay ** (j - i - 1))

    prerequisite_penalty = 0.0
    violated = 0
    for lesson_a, lesson_b, edge_weight in prerequisite_edges:
        if position[lesson_a] > position[lesson_b]:
            violated += 1
            prerequisite_penalty += prerequisite_weight * edge_weight * (position[lesson_a] - position[lesson_b] + 1)

    coherence = similarity_reward - prerequisite_penalty

    if return_components:
        return {
            "coherence": float(coherence),
            "similarity_reward": float(similarity_reward),
            "prerequisite_penalty": float(prerequisite_penalty),
            "violated_prerequisites": float(violated),
        }

    return float(coherence)


def _neighbor(order: Sequence[LessonId], rng: random.Random, swap_rate: float) -> list[LessonId]:
    # Генерирует соседнее решение для simulated annealing через swap или перенос одного занятия.
    order = list(order)
    if len(order) < 2:
        return order

    i, j = rng.sample(range(len(order)), 2)

    if rng.random() < swap_rate:
        order[i], order[j] = order[j], order[i]
        return order

    item = order.pop(i)
    if j > i:
        j -= 1
    order.insert(j, item)
    return order


def optimize_lesson_order_sa(
    initial_order: Sequence[LessonId],
    lesson_similarity: dict[str, Any],
    lesson_prerequisite_edges: dict[str, Any] | Sequence[tuple[LessonId, LessonId] | tuple[LessonId, LessonId, float]] | None = None,
    *,
    itermax: int = 50_000,
    T0: float = 10.0,
    Tmin: float = 0.05,
    cooling_rate: float = 0.99,
    rho: float = 0.04,
    swap_rate: float = 0.43,
    similarity_weight: float = 1.0,
    closeness_decay: float = 0.85,
    prerequisite_weight: float = 10.0,
    max_similarity_distance: int | None = None,
    seed: int | None = None,
    keep_history_every: int = 0,
) -> dict[str, Any]:
    # Ищет улучшенный порядок занятий методом simulated annealing по функции coherence.
    rng = random.Random(seed)

    def cost(order: Sequence[LessonId]) -> float:
        return -float(
            evaluate_lesson_order_coherence(
                lesson_order=order,
                lesson_similarity=lesson_similarity,
                lesson_prerequisite_edges=lesson_prerequisite_edges,
                similarity_weight=similarity_weight,
                closeness_decay=closeness_decay,
                prerequisite_weight=prerequisite_weight,
                max_similarity_distance=max_similarity_distance,
            )
        )

    current_order = list(initial_order)
    current_cost = cost(current_order)
    best_order = list(current_order)
    best_cost = float(current_cost)

    num_levels = 1 if T0 == Tmin else max(1, math.ceil(math.log(Tmin / T0) / math.log(cooling_rate)))
    ns = max(1, int(round(itermax / num_levels)))
    na = max(1, min(ns, math.ceil(rho * ns)))

    T = float(T0)
    iterations_done = 0
    accepted_moves = 0

    while iterations_done < itermax:
        sampled = 0
        accepted_at_temp = 0

        while iterations_done < itermax and sampled < ns:
            candidate_order = _neighbor(current_order, rng, swap_rate)
            candidate_cost = cost(candidate_order)
            delta = candidate_cost - current_cost

            if delta <= 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                current_order = candidate_order
                current_cost = candidate_cost
                accepted_moves += 1
                accepted_at_temp += 1

                if current_cost < best_cost:
                    best_order = list(current_order)
                    best_cost = float(current_cost)

            sampled += 1
            iterations_done += 1

            if accepted_at_temp >= na:
                break

        if T > Tmin:
            T = max(Tmin, T * cooling_rate)

    return {
        "best_order": best_order,
        "best_coherence": float(-best_cost),
        "best_cost": float(best_cost),
        "iterations_done": int(iterations_done),
        "accepted_moves": int(accepted_moves),
        "temperature_final": float(T),
        "ns": int(ns),
        "na": int(na),
    }
