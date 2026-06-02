from __future__ import annotations

import numpy as np


def compare_lesson_topic_profiles(
    lesson_a_profile: dict[int, float],
    lesson_b_profile: dict[int, float],
) -> float:
    # Суммирует минимальные веса по общим темам двух занятий.
    if not lesson_a_profile or not lesson_b_profile:
        return 0.0

    common_topics = set(lesson_a_profile) & set(lesson_b_profile)
    return float(
        sum(
            min(float(lesson_a_profile[topic_id]), float(lesson_b_profile[topic_id]))
            for topic_id in common_topics
        )
    )


def build_lesson_similarity_matrix(
    lesson_topic_profiles: dict[str, dict[int, float]],
) -> dict[str, object]:
    # Строит симметричную матрицу схожести всех занятий.
    if not lesson_topic_profiles:
        raise ValueError("lesson_topic_profiles пуст.")

    lesson_names = list(lesson_topic_profiles)
    n_lessons = len(lesson_names)
    similarity_matrix = np.eye(n_lessons, dtype=float)

    for i in range(n_lessons):
        for j in range(i + 1, n_lessons):
            score = compare_lesson_topic_profiles(
                lesson_topic_profiles[lesson_names[i]],
                lesson_topic_profiles[lesson_names[j]],
            )
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score

    return {
        "lesson_names": lesson_names,
        "similarity_matrix": similarity_matrix,
    }
