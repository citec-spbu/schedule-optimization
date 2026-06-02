import math
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


COURSES_TOPICS_FILE = "courses_topics.json"
SELECTED_CONCEPTS_FILE = "selected_concepts.json"

CSR_OUTPUT_FILE = "csr_scores.csv"
MISSING_TOPICS_FILE = "missing_topics.csv"

GRAPHS_OUTPUT_DIR = "ace_graphs"

NGRAM_SIZE = 10
NGRAM_STEP = 10
TOP_T_PERCENT = 100.0

RUN_INTERACTIVE_ACE = True

CALCULATE_CER = False

# None = обработать все курсы из selected_concepts.json.
COURSES_TO_PROCESS = None

BATCH_SIZE = 32
SHOW_PROGRESS_BAR = True

DEFAULT_MODEL = SentenceTransformer(MODEL_NAME)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    Нормирует строки матрицы по L2-норме.
    Нужно для вычисления cosine similarity через скалярное произведение.
    """
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def normalize_vector(x: np.ndarray) -> np.ndarray:
    """
    Нормирует один вектор по L2-норме.
    """
    x = np.asarray(x)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def make_ngrams(text: str, n: int = 10, step: int = 10) -> list[str]:
    """
    Разбивает текст на n-граммы по словам.
    """
    words = str(text).split()

    if not words:
        return [""]

    if len(words) <= n:
        return [" ".join(words)]

    return [
        " ".join(words[i:i + n])
        for i in range(0, len(words) - n + 1, step)
    ]


def normalize_text_for_cer(text: str) -> str:
    """
    Нормализует текст для поиска точного вхождения
    """
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cer(concept: str, chunks: list[str]) -> int:
    """
    CER = число n-грамм описания, в которых напрямую встречается название понятия.

    В основном запуске CER не используется. Функция оставлена как опция,
    поскольку CER описан в теоретической части алгоритма как альтернатива CSR.
    """
    concept = normalize_text_for_cer(concept)

    if not concept:
        return 0

    count = 0

    for chunk in chunks:
        chunk = normalize_text_for_cer(chunk)

        if concept in chunk:
            count += 1

    return count


def csr_from_embeddings(
    concept_emb: np.ndarray,
    chunk_embs: np.ndarray,
) -> float:
    """
    CSR = сумма cosine similarity между эмбеддингом названия понятия
    и эмбеддингами всех n-грамм описания.
    """
    concept_emb = normalize_vector(concept_emb)
    chunk_embs = normalize_rows(chunk_embs)

    sims = chunk_embs @ concept_emb
    return float(sims.sum())



def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_courses(
    data: dict,
    courses_to_process: list[str] | None = None,
) -> dict:
    """
    Оставляет только выбранные курсы, если courses_to_process не None.
    """
    if courses_to_process is None:
        return data

    return {
        course: data[course]
        for course in courses_to_process
        if course in data
    }


def build_topic_texts_by_course(
    courses_topics: dict,
    selected_concepts: dict,
    courses_to_process: list[str] | None = None,
) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    """
    Собирает для выбранных понятий их заранее подготовленные описания.

    courses_topics.json имеет структуру:
        курс -> понятие -> описание

    selected_concepts.json имеет структуру:
        курс -> список понятий

    Если для выбранного понятия нет готового описания в courses_topics.json,
    оно записывается в missing_topics.csv.
    """
    selected_concepts = filter_courses(
        selected_concepts,
        courses_to_process=courses_to_process,
    )

    result = {}
    missing_rows = []

    for course, selected_topics in selected_concepts.items():
        if course not in courses_topics:
            missing_rows.append({
                "course": course,
                "topic": "",
                "reason": "course_not_found_in_courses_topics"
            })
            continue

        course_descriptions = courses_topics[course]
        topic_to_text = {}

        for topic in selected_topics:
            if topic not in course_descriptions:
                missing_rows.append({
                    "course": course,
                    "topic": topic,
                    "reason": "topic_description_not_found"
                })
                continue

            description = str(course_descriptions[topic]).strip()

            if not description:
                missing_rows.append({
                    "course": course,
                    "topic": topic,
                    "reason": "empty_description"
                })
                continue

            topic_to_text[topic] = description

        if topic_to_text:
            result[course] = topic_to_text

    missing_df = pd.DataFrame(
        missing_rows,
        columns=["course", "topic", "reason"]
    )

    return result, missing_df

def prepare_embeddings_for_course(
    topic_to_text: dict[str, str],
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    """
    Предварительно считает эмбеддинги
    """
    topics = list(topic_to_text.keys())

    topic_vectors = model.encode(
        topics,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    topic_embs = {
        topic: topic_vectors[i]
        for i, topic in enumerate(topics)
    }

    topic_chunks = {
        topic: make_ngrams(text, n=n, step=step)
        for topic, text in topic_to_text.items()
    }

    all_chunks = []
    chunk_owner = []

    for topic, chunks in topic_chunks.items():
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_owner.append(topic)

    all_chunk_vectors = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    chunk_embs = {topic: [] for topic in topics}

    for owner, vector in zip(chunk_owner, all_chunk_vectors):
        chunk_embs[owner].append(vector)

    chunk_embs = {
        topic: np.asarray(vectors)
        for topic, vectors in chunk_embs.items()
    }

    return topic_embs, chunk_embs, topic_chunks


def calculate_csr_for_course(
    course: str,
    topic_to_text: dict[str, str],
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
    batch_size: int = 32,
    show_progress_bar: bool = True,
    calculate_cer: bool = CALCULATE_CER,
) -> tuple[pd.DataFrame, list[tuple[float, str, str]], dict[tuple[str, str], tuple[float, float]]]:
    """
    Считает CSR для всех неупорядоченных пар понятий внутри одного курса.

    Для пары (A, B) считаются два значения:

    csr_description_a_to_concept_b:
        насколько описание A семантически связано с названием B

    csr_description_b_to_concept_a:
        насколько описание B семантически связано с названием A

    PRS = max(этих двух значений)

    Если calculate_cer=True, дополнительно считаются и сохраняются CER-значения.
    По умолчанию CER не считается.

    Возвращает:
    1. DataFrame с CSR-таблицей.
    2. ranked_pairs для ACE-алгоритма.
    3. pair_scores для ACE-алгоритма.
    """
    topics = list(topic_to_text.keys())

    topic_embs, topic_chunk_embs, topic_chunks = prepare_embeddings_for_course(
        topic_to_text=topic_to_text,
        model=model,
        n=n,
        step=step,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    rows = []
    ranked_pairs = []
    pair_scores = {}

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            topic_a = topics[i]
            topic_b = topics[j]

            score_ab = csr_from_embeddings(
                concept_emb=topic_embs[topic_b],
                chunk_embs=topic_chunk_embs[topic_a],
            )

            score_ba = csr_from_embeddings(
                concept_emb=topic_embs[topic_a],
                chunk_embs=topic_chunk_embs[topic_b],
            )

            prs = max(score_ab, score_ba)

            if score_ab >= score_ba:
                max_csr_direction = f"description({topic_a}) -> concept({topic_b})"
            else:
                max_csr_direction = f"description({topic_b}) -> concept({topic_a})"

            row = {
                "course": course,
                "topic_a": topic_a,
                "topic_b": topic_b,
                "csr_description_a_to_concept_b": score_ab,
                "csr_description_b_to_concept_a": score_ba,
                "prs": prs,
                "max_csr_direction": max_csr_direction,
                "chunks_count_a": len(topic_chunks[topic_a]),
                "chunks_count_b": len(topic_chunks[topic_b]),
                "ngram_size": n,
                "ngram_step": step,
                "model": MODEL_NAME,
            }

            if calculate_cer:
                cer_ab = cer(
                    concept=topic_b,
                    chunks=topic_chunks[topic_a],
                )

                cer_ba = cer(
                    concept=topic_a,
                    chunks=topic_chunks[topic_b],
                )

                cer_prs = max(cer_ab, cer_ba)

                if cer_ab >= cer_ba:
                    max_cer_direction = f"description({topic_a}) -> concept({topic_b})"
                else:
                    max_cer_direction = f"description({topic_b}) -> concept({topic_a})"

                row.update({
                    "cer_description_a_to_concept_b": cer_ab,
                    "cer_description_b_to_concept_a": cer_ba,
                    "cer_prs": cer_prs,
                    "max_cer_direction": max_cer_direction,
                })

            rows.append(row)

            ranked_pairs.append((prs, topic_a, topic_b))
            pair_scores[(topic_a, topic_b)] = (score_ab, score_ba)

    ranked_pairs.sort(reverse=True, key=lambda x: x[0])

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["course", "prs"],
            ascending=[True, False],
        )

    return df, ranked_pairs, pair_scores


def calculate_and_save_all_csr(
    topic_texts_by_course: dict[str, dict[str, str]],
    output_file: str | Path = CSR_OUTPUT_FILE,
    model: SentenceTransformer = DEFAULT_MODEL,
    n: int = 10,
    step: int = 10,
    batch_size: int = 32,
    show_progress_bar: bool = True,
    calculate_cer: bool = CALCULATE_CER,
) -> tuple[pd.DataFrame, dict]:
    """
    Считает CSR по всем курсам и сохраняет общий CSV-файл.

    Возвращает:
    1. combined_df — общая таблица CSR.
    2. precomputed_by_course — данные для последующего запуска ACE без повторного расчёта CSR.
    """
    all_dfs = []
    precomputed_by_course = {}

    for course, topic_to_text in topic_texts_by_course.items():
        print("\n" + "=" * 80)
        print(f"Расчёт CSR для курса: {course}")
        print(f"Количество понятий: {len(topic_to_text)}")

        if calculate_cer:
            print("CER также будет рассчитан и добавлен в CSV.")

        print("=" * 80)

        csr_df, ranked_pairs, pair_scores = calculate_csr_for_course(
            course=course,
            topic_to_text=topic_to_text,
            model=model,
            n=n,
            step=step,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            calculate_cer=calculate_cer,
        )

        all_dfs.append(csr_df)

        precomputed_by_course[course] = {
            "topic_to_text": topic_to_text,
            "ranked_pairs": ranked_pairs,
            "pair_scores": pair_scores,
        }

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.sort_values(
            by=["course", "prs"],
            ascending=[True, False],
        )
    else:
        combined_df = pd.DataFrame()

    output_file = Path(output_file)
    combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"CSR сохранён в файл: {output_file.resolve()}")
    print(f"Всего строк с парами понятий: {len(combined_df)}")
    print("=" * 80)

    return combined_df, precomputed_by_course


# ======================================================================
# ========================== ACE-АЛГОРИТМ ===============================
# ======================================================================

def ask_expert(
    topic_a: str,
    topic_b: str,
    score_ab: float,
    score_ba: float,
    prs: float,
) -> int:
    """
    Интерактивный запрос решения эксперта.
    """
    print("\n" + "=" * 80)
    print(f"Пара: {topic_a!r}  <->  {topic_b!r}")
    print(f"CSR(description({topic_a}) -> concept({topic_b})) = {score_ab:.6f}")
    print(f"CSR(description({topic_b}) -> concept({topic_a})) = {score_ba:.6f}")
    print(f"PRS = {prs:.6f}")
    print("Введите решение:")
    print(f"  0 -> {topic_a} prerequisite для {topic_b}   ({topic_a} -> {topic_b})")
    print(f"  1 -> {topic_b} prerequisite для {topic_a}   ({topic_b} -> {topic_a})")
    print("  2 -> остановить разметку для текущего курса")

    while True:
        ans = input("Ваш выбор [0/1/2]: ").strip()

        if ans in {"0", "1", "2"}:
            return int(ans)

        print("Некорректный ввод. Нужно ввести 0, 1 или 2.")


def run_ace_from_precomputed_csr(
    topic_to_text: dict[str, str],
    ranked_pairs: list[tuple[float, str, str]],
    pair_scores: dict[tuple[str, str], tuple[float, float]],
    top_t_percent: float = 100.0,
) -> nx.DiGraph:
    """
    Запускает ACE-алгоритм уже на заранее рассчитанных CSR.
    """
    g = nx.DiGraph()
    g.add_nodes_from(topic_to_text.keys())

    limit = math.ceil(len(ranked_pairs) * top_t_percent / 100)

    for prs, topic_a, topic_b in ranked_pairs[:limit]:
        # Если между вершинами уже есть путь в любую сторону, пара пропускается, чтобы не нарушать минимальность и не создавать цикл.
        if nx.has_path(g, topic_a, topic_b) or nx.has_path(g, topic_b, topic_a):
            continue

        score_ab, score_ba = pair_scores[(topic_a, topic_b)]

        ans = ask_expert(
            topic_a=topic_a,
            topic_b=topic_b,
            score_ab=score_ab,
            score_ba=score_ba,
            prs=prs,
        )

        if ans == 2:
            break

        if ans == 0:
            g.add_edge(topic_a, topic_b)
        else:
            g.add_edge(topic_b, topic_a)

        # После добавления ребра оставляем транзитивно сокращённый граф.
        g = nx.transitive_reduction(g)

    return g


# ======================================================================
# ============================= СОХРАНЕНИЕ ГРАФА =========================
# ======================================================================

def safe_filename(name: str) -> str:
    """
    Делает безопасное имя файла из названия курса.
    """
    name = str(name).strip()
    name = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def save_graph(
    g: nx.DiGraph,
    course: str,
    output_dir: str | Path = GRAPHS_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """
    Сохраняет граф курса в два файла:
    1. nodes CSV
    2. edges CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    course_name = safe_filename(course)

    nodes_file = output_dir / f"{course_name}_nodes.csv"
    edges_file = output_dir / f"{course_name}_edges.csv"

    pd.DataFrame({
        "concept": list(g.nodes())
    }).to_csv(nodes_file, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        list(g.edges()),
        columns=["source", "target"]
    ).to_csv(edges_file, index=False, encoding="utf-8-sig")

    return nodes_file, edges_file


def print_graph(g: nx.DiGraph, course: str) -> None:
    print("\n" + "=" * 80)
    print(f"Итоговые ребра графа для курса: {course}")

    if g.number_of_edges() == 0:
        print("Рёбер нет.")
    else:
        for u, v in g.edges():
            print(f"{u} -> {v}")

    print("=" * 80)


# ======================================================================
# ============================== ОСНОВНОЙ ЗАПУСК =========================
# ======================================================================

def run_from_json_files(
    courses_topics_file: str | Path = COURSES_TOPICS_FILE,
    selected_concepts_file: str | Path = SELECTED_CONCEPTS_FILE,
    csr_output_file: str | Path = CSR_OUTPUT_FILE,
    missing_topics_file: str | Path = MISSING_TOPICS_FILE,
    graphs_output_dir: str | Path = GRAPHS_OUTPUT_DIR,
    courses_to_process: list[str] | None = COURSES_TO_PROCESS,
    run_interactive_ace: bool = RUN_INTERACTIVE_ACE,
    calculate_cer: bool = CALCULATE_CER,
) -> None:
    """
    Основной сценарий:

    1. Загрузить courses_topics.json с готовыми описаниями понятий.
    2. Загрузить selected_concepts.json со списком обрабатываемых понятий.
    3. Собрать готовые описания для выбранных понятий.
    4. Предварительно рассчитать CSR.
    5. Сохранить CSR в отдельный CSV-файл.
    6. После этого, при необходимости, запустить ACE-алгоритм.
    """
    courses_topics = load_json(courses_topics_file)
    selected_concepts = load_json(selected_concepts_file)

    topic_texts_by_course, missing_df = build_topic_texts_by_course(
        courses_topics=courses_topics,
        selected_concepts=selected_concepts,
        courses_to_process=courses_to_process,
    )

    if not missing_df.empty:
        missing_topics_file = Path(missing_topics_file)
        missing_df.to_csv(missing_topics_file, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 80)
        print(f"Есть темы без найденных описаний. Они сохранены в файл: {missing_topics_file.resolve()}")
        print("=" * 80)

    if not topic_texts_by_course:
        raise ValueError("Не найдено ни одного курса с описаниями выбранных понятий.")

    csr_df, precomputed_by_course = calculate_and_save_all_csr(
        topic_texts_by_course=topic_texts_by_course,
        output_file=csr_output_file,
        model=DEFAULT_MODEL,
        n=NGRAM_SIZE,
        step=NGRAM_STEP,
        batch_size=BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS_BAR,
        calculate_cer=calculate_cer,
    )

    if not run_interactive_ace:
        print("\nИнтерактивный ACE-алгоритм не запущен, потому что RUN_INTERACTIVE_ACE = False.")
        return

    for course, data in precomputed_by_course.items():
        print("\n" + "#" * 80)
        print(f"Запуск ACE-алгоритма для курса: {course}")
        print("#" * 80)

        graph = run_ace_from_precomputed_csr(
            topic_to_text=data["topic_to_text"],
            ranked_pairs=data["ranked_pairs"],
            pair_scores=data["pair_scores"],
            top_t_percent=TOP_T_PERCENT,
        )

        nodes_file, edges_file = save_graph(
            g=graph,
            course=course,
            output_dir=graphs_output_dir,
        )

        print_graph(graph, course=course)

        print(f"Файл вершин: {nodes_file.resolve()}")
        print(f"Файл рёбер: {edges_file.resolve()}")


if __name__ == "__main__":
    run_from_json_files()
