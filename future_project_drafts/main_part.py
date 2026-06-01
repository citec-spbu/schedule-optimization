from __future__ import annotations

import numpy as np

from text_preprocessing import preprocess_lesson_transcript
from text_segmentation import select_global_c, segment_with_embed_kcpd
from topic_modeling import topicize_segmented_corpus
from lessons_similarity import build_lesson_similarity_matrix
from prerequsite_relations import (
    build_topic_payloads,
    collect_topic_prerequisite_results,
)
from coherence_function import (
    build_lesson_prerequisite_edges_from_topic_results,
    evaluate_lesson_order_coherence,
    optimize_lesson_order_sa,
)


TOPIC_STOP_WORDS = [
    "это", "этот", "эта", "эти", "того", "такая", "такой", "такие",
    "который", "которая", "которые", "которое", "также", "затем", "далее",
    "после", "этого", "сначала", "потом", "здесь", "там", "теперь", "итак",
    "итого", "можно", "нужно", "надо", "часто", "лишь", "уже", "еще",
    "именно", "просто", "либо", "если", "при", "для", "про", "об", "под",
    "над", "между", "через", "перед", "внутри", "вместе", "блок", "часть",
    "тема", "занятие", "лекция", "раздел", "фрагмент", "материал", "пример",
    "примеры", "случай", "случаи", "метод", "методы", "способ", "способы",
    "обсуждаем", "рассматриваем", "вводим", "вводится", "разбираем",
    "изучаем", "говорим", "переходим", "посвящен", "посвящена", "относится",
    "связан", "связана", "получается", "является", "являются", "на",
]


def build_demo_schedule() -> list[dict[str, str]]:
    # Создает небольшое искусственное расписание с текстами занятий для демонстрации всего пайплайна.
    return [
        {
            "lesson_name": "mixed_control_example",
            "text": """
            Реляционная база данных хранит информацию в таблицах.
            Первичный ключ однозначно идентифицирует запись.
            Внешний ключ задает связь между таблицами.
            SQL запрос выполняет выборку, соединение и агрегацию данных.
            Нормализация уменьшает избыточность и аномалии обновления.
            Клетка является базовой структурной единицей живого организма.
            Ядро хранит генетическую информацию.
            Митохондрии участвуют в энергетическом обмене.
            Рибосомы синтезируют белок.
            Клеточная мембрана регулирует транспорт веществ.
            Республика в Древнем Риме опиралась на сенат и выборные магистратуры.
            Юлий Цезарь усилил личную власть в период политического кризиса.
            Октавиан Август оформил переход к империи.
            Гражданские войны изменили римскую политическую систему.
            Классификация использует размеченные данные и дискретные классы.
            Регрессия предсказывает непрерывную числовую величину.
            Переобучение возникает при чрезмерной подстройке модели под обучающую выборку.
            Регуляризация снижает сложность модели и улучшает обобщающую способность.
            """,
        },
        {
            "lesson_name": "classical_mechanics",
            "text": """
            Система отсчета и координаты задают положение тела в пространстве.
            Средняя скорость определяется через изменение координаты за интервал времени.
            Мгновенная скорость получается как предел средней скорости.
            Ускорение описывает изменение скорости во времени.
            Формулы равноускоренного движения связывают координату, скорость и время.
            Первый закон Ньютона описывает инерциальное движение без результирующей силы.
            Второй закон Ньютона связывает силу, массу и ускорение.
            Третий закон Ньютона выражает действие и противодействие.
            Сила тяжести, реакция опоры и натяжение нити входят в базовые задачи динамики.
            Работа силы зависит от перемещения и угла между силой и направлением движения.
            Кинетическая энергия определяется через массу и квадрат скорости.
            Потенциальная энергия связана с положением в поле сил.
            Закон сохранения механической энергии упрощает решение задач без явного интегрирования уравнений.
            Импульс равен произведению массы на скорость.
            Закон сохранения импульса применяется к столкновениям и замкнутым системам.
            Упругий удар и неупругий удар различаются по сохранению кинетической энергии.
            """,
        },
        {
            "lesson_name": "mathematical_analysis",
            "text": """
            Предел последовательности определяется через epsilon окрестность и номер N.
            Сходящаяся последовательность имеет единственный предел.
            Предел суммы равен сумме пределов.
            Предел произведения выражается через пределы множителей.
            Предел функции в точке определяется по Коши.
            Определение по Гейне использует последовательности аргументов.
            Односторонний предел описывает поведение функции слева и справа.
            Бесконечный предел и предел на бесконечности расширяют стандартную схему.
            Непрерывность функции в точке означает равенство предела и значения.
            Разрывы делятся на устранимые, скачки и разрывы второго рода.
            Теорема Вейерштрасса гарантирует достижение максимума и минимума.
            Теорема о промежуточном значении описывает прохождение всех промежуточных значений.
            Производная определяется как предел отношения приращений.
            Геометрический смысл производной связан с касательной.
            Физический смысл производной связан с мгновенной скоростью.
            Правила дифференцирования включают сумму, произведение и сложную функцию.
            """,
        },
        {
            "lesson_name": "algorithms_and_data_structures",
            "text": """
            Асимптотическая сложность описывает рост времени работы алгоритма.
            Нотация O большое используется для верхней оценки сложности.
            Линейная сложность характерна для последовательного просмотра массива.
            Квадратичная сложность часто возникает у простых сортировок.
            Логарифмическая сложность появляется в двоичном поиске.
            Массив обеспечивает быстрый доступ по индексу.
            Связный список удобен для вставок и удалений.
            Стек реализует дисциплину LIFO.
            Очередь реализует дисциплину FIFO.
            Хеш таблица хранит пары ключ значение и ускоряет поиск.
            Дерево поиска поддерживает упорядоченное хранение элементов.
            Рекурсия описывает задачу через более простой экземпляр той же задачи.
            Divide and conquer разбивает задачу на подзадачи и объединяет ответы.
            Сортировка слиянием использует рекурсивное разбиение и этап merge.
            Граф состоит из вершин и ребер.
            Обход в ширину и обход в глубину решают разные задачи на графах.
            """,
        },
    ]


def main() -> None:
    # Запускает весь демонстрационный пайплайн: предобработка, сегментация, тематизация,
    # построение связей между темами и занятиями, оценка расписания и его оптимизация.
    schedule = build_demo_schedule()
    texts = {x["lesson_name"]: x["text"] for x in schedule}
    initial_order = [x["lesson_name"] for x in schedule]

    def print_report(title: str, order: list[str], lesson_similarity: dict, lesson_edges: dict) -> None:
        # Печатает краткий отчет по текущему порядку занятий и его значению функции согласованности.
        score = evaluate_lesson_order_coherence(
            lesson_order=order,
            lesson_similarity=lesson_similarity,
            lesson_prerequisite_edges=lesson_edges,
            similarity_weight=1.0,
            closeness_decay=0.85,
            prerequisite_weight=10.0,
            return_components=True,
        )
        print(f"\n=== {title} ===")
        print("Порядок:")
        for i, name in enumerate(order, 1):
            print(f"{i}. {name}")
        print(f"coherence={score['coherence']:.4f}")
        print(f"similarity_reward={score['similarity_reward']:.4f}")
        print(f"prerequisite_penalty={score['prerequisite_penalty']:.4f}")
        print(f"violated_prerequisites={int(score['violated_prerequisites'])}")

    print("=== ИСХОДНОЕ РАСПИСАНИЕ ===")
    for i, name in enumerate(initial_order, 1):
        print(f"{i}. {name}")

    processed = {
        lesson_name: preprocess_lesson_transcript(text=text, min_sentence_len=15)
        for lesson_name, text in texts.items()
    }

    c_info = select_global_c(
        corpus_embeddings=[x["sentence_embeddings"] for x in processed.values()],
        kernel="cosine",
        aggregation="median",
        min_size=2,
        normalize_embeddings=True,
        random_state=42,
    )
    chosen_c = float(c_info["global_c"])
    print(f"\nВыбран global_c: {chosen_c:.8f}")

    segmented_embeddings: dict[str, list[np.ndarray]] = {}
    segmented_texts: dict[str, list[str]] = {}

    print("\n=== СЕГМЕНТАЦИЯ ===")
    for lesson_name, item in processed.items():
        seg = segment_with_embed_kcpd(
            sentence_embeddings=item["sentence_embeddings"],
            c=chosen_c,
            kernel="cosine",
            min_size=2,
            normalize_embeddings=True,
        )

        lesson_embs = [item["sentence_embeddings"][start:end] for start, end in seg["segments"]]
        lesson_texts = [" ".join(item["sentences"][start:end]) for start, end in seg["segments"]]

        segmented_embeddings[lesson_name] = lesson_embs
        segmented_texts[lesson_name] = lesson_texts

        print(f"{lesson_name}: {len(seg['segments'])} сегм.")
        for i, text in enumerate(lesson_texts, 1):
            first_sentence = text.split(".")[0].strip()
            print(f"  {i}. {first_sentence[:120]}")

    topic_results = topicize_segmented_corpus(
        segmented_embeddings=segmented_embeddings,
        segmented_texts=segmented_texts,
        aggregation="mean",
        normalize_segment_vectors=True,
        min_topic_size=2,
        top_n_words=10,
        ngram_range=(1, 3),
        stop_words=TOPIC_STOP_WORDS,
        language="multilingual",
        calculate_probabilities=False,
        nr_topics=None,
        umap_n_neighbors=5,
        umap_n_components=5,
        umap_min_dist=0.0,
        umap_metric="cosine",
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        lesson_profile_weight_mode="token_count",
        ignore_outliers_in_profiles=False,
        random_state=42,
    )

    print("\n=== ТЕМЫ ===")
    for topic_id, info in sorted(topic_results["topic_info"].items()):
        keywords = ", ".join(word for word, _ in info["keywords"][:5]) if info["keywords"] else "-"
        print(f"topic_{topic_id}: {info['topic_name']} | size={info['size']} | keywords={keywords}")

    print("\n=== ТЕМЫ ПО ЗАНЯТИЯМ ===")
    for lesson_name, profile in topic_results["lesson_topic_profiles"].items():
        parts = [
            f"topic_{topic_id}={weight:.3f}"
            for topic_id, weight in sorted(profile.items(), key=lambda x: x[1], reverse=True)
        ]
        print(f"{lesson_name}: " + (", ".join(parts) if parts else "-"))

    similarity_results = build_lesson_similarity_matrix(
        lesson_topic_profiles=topic_results["lesson_topic_profiles"]
    )

    print("\n=== ПОПАРНАЯ СХОЖЕСТЬ ЗАНЯТИЙ ===")
    lesson_names = similarity_results["lesson_names"]
    sim_matrix = similarity_results["similarity_matrix"]
    for i in range(len(lesson_names)):
        for j in range(i + 1, len(lesson_names)):
            print(f"{lesson_names[i]} <-> {lesson_names[j]} : {sim_matrix[i, j]:.4f}")

    topic_payloads = build_topic_payloads(
        topic_results=topic_results,
        segmented_embeddings=segmented_embeddings,
        ignore_outliers=True,
    )

    topic_prerequisite_results = collect_topic_prerequisite_results(
        topic_payloads=topic_payloads,
        local_unit="segment",
        aggregation="mean",
        prototype_mode="all_sentences_mean",
        min_prs_mean=0.35,
        min_direction_margin=0.06,
        cer_weight=0.10,
    )

    directed = [
        x for x in topic_prerequisite_results
        if x["decision"]["direction"] in {"a->b", "b->a"}
    ]
    directed.sort(
        key=lambda x: (x["csr"]["prs_mean"], x["direction_scores"]["direction_margin"]),
        reverse=True,
    )

    print("\n=== PREREQUISITE-СВЯЗИ МЕЖДУ ТЕМАМИ ===")
    if directed:
        for i, x in enumerate(directed, 1):
            if x["decision"]["direction"] == "a->b":
                a_id, a_name, b_id, b_name = (
                    x["topic_a_id"],
                    x["topic_a_name"],
                    x["topic_b_id"],
                    x["topic_b_name"],
                )
            else:
                a_id, a_name, b_id, b_name = (
                    x["topic_b_id"],
                    x["topic_b_name"],
                    x["topic_a_id"],
                    x["topic_a_name"],
                )

            print(
                f"{i}. topic_{a_id} ({a_name}) -> topic_{b_id} ({b_name}) | "
                f"prs_mean={x['csr']['prs_mean']:.4f} | "
                f"margin={x['direction_scores']['direction_margin']:.4f}"
            )
    else:
        print("Направленные связи не найдены.")

    lesson_prerequisite_info = build_lesson_prerequisite_edges_from_topic_results(
        lesson_topic_profiles=topic_results["lesson_topic_profiles"],
        prerequisite_results=topic_prerequisite_results,
        topic_edge_weight_mode="prs_mean",
        lesson_edge_weight_mode="product",
        min_prs_mean=0.35,
        min_direction_margin=0.06,
        min_topic_weight_in_lesson=0.0,
        drop_self_edges=True,
    )

    print("\n=== PREREQUISITE-СВЯЗИ МЕЖДУ ЗАНЯТИЯМИ ===")
    if lesson_prerequisite_info["lesson_edges"]:
        for i, (a, b, w) in enumerate(lesson_prerequisite_info["lesson_edges"][:20], 1):
            print(f"{i}. {a} -> {b} | weight={w:.4f}")
    else:
        print("Связи не получены.")

    print_report(
        "ИСХОДНОЕ РАСПИСАНИЕ",
        initial_order,
        similarity_results,
        lesson_prerequisite_info,
    )

    opt = optimize_lesson_order_sa(
        initial_order=initial_order,
        lesson_similarity=similarity_results,
        lesson_prerequisite_edges=lesson_prerequisite_info,
        itermax=20000,
        T0=10.0,
        Tmin=0.05,
        cooling_rate=0.99,
        rho=0.04,
        swap_rate=0.43,
        similarity_weight=1.0,
        closeness_decay=0.85,
        prerequisite_weight=10.0,
        max_similarity_distance=None,
        seed=42,
        keep_history_every=0,
    )

    improved_order = opt["best_order"]

    print_report(
        "УЛУЧШЕННОЕ РАСПИСАНИЕ",
        improved_order,
        similarity_results,
        lesson_prerequisite_info,
    )

    print("\n=== РЕЗУЛЬТАТ ОПТИМИЗАЦИИ ===")
    print(f"best_cost={opt['best_cost']:.4f}")
    print(f"best_coherence={opt['best_coherence']:.4f}")
    print(f"iterations_done={opt['iterations_done']}")
    print(f"accepted_moves={opt['accepted_moves']}")
    print(f"temperature_final={opt['temperature_final']:.4f}")
    print(f"ns={opt['ns']}")
    print(f"na={opt['na']}")


if __name__ == "__main__":
    main()
