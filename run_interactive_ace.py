import json
import math
import re
from pathlib import Path

import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "extracted_data"

CSR_INPUT_FILE = DATA_DIR / "csr_scores.csv"
SELECTED_CONCEPTS_FILE = DATA_DIR / "selected_concepts.json"

GRAPHS_OUTPUT_DIR = BASE_DIR / "ace_graphs"

TOP_T_PERCENT = 100.0

# None = запустить интерактивный ACE для всех курсов из csr_scores.csv.
COURSES_TO_PROCESS = None


REQUIRED_CSR_COLUMNS = [
    "course",
    "topic_a",
    "topic_b",
    "csr_description_a_to_concept_b",
    "csr_description_b_to_concept_a",
    "prs",
]

def load_json(path: str | Path) -> dict:
    """
    Загружает JSON-файл.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csr_scores(path: str | Path = CSR_INPUT_FILE) -> pd.DataFrame:
    """
    Загружает заранее рассчитанный csr_scores.csv.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл CSR не найден: {path}. "
            f"Сначала запустите calculate_csr.py."
        )

    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_CSR_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "В csr_scores.csv не хватает обязательных колонок: "
            + ", ".join(missing_columns)
        )

    df = df.sort_values(
        by=["course", "prs"],
        ascending=[True, False],
    )

    return df


def filter_courses(
    courses: list[str],
    courses_to_process: list[str] | None = None,
) -> list[str]:
    """
    Оставляет только выбранные курсы, если courses_to_process не None.
    """
    if courses_to_process is None:
        return courses

    return [
        course
        for course in courses
        if course in courses_to_process
    ]


def safe_filename(name: str) -> str:
    """
    Делает безопасное имя файла из названия курса.
    """
    name = str(name).strip()
    name = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def get_course_topics_from_selected_concepts(
    course: str,
    selected_concepts_file: str | Path = SELECTED_CONCEPTS_FILE,
) -> list[str] | None:
    """
    Пытается получить полный список понятий курса из selected_concepts.json.
    Если файл отсутствует или курса в нём нет, возвращает None.
    """
    selected_concepts_file = Path(selected_concepts_file)

    if not selected_concepts_file.exists():
        return None

    selected_concepts = load_json(selected_concepts_file)

    if course not in selected_concepts:
        return None

    return list(selected_concepts[course])


def get_course_topics_from_csr(course_df: pd.DataFrame) -> list[str]:
    """
    Восстанавливает список понятий курса из csr_scores.csv.
    """
    topics = set(course_df["topic_a"]).union(set(course_df["topic_b"]))
    return sorted(topics)


def get_course_topics(
    course: str,
    course_df: pd.DataFrame,
    selected_concepts_file: str | Path = SELECTED_CONCEPTS_FILE,
) -> list[str]:
    """
    Возвращает список понятий курса.

    Приоритет:
    1. selected_concepts.json — сохраняет исходный порядок понятий;
    2. csr_scores.csv — резервный вариант.
    """
    selected_topics = get_course_topics_from_selected_concepts(
        course=course,
        selected_concepts_file=selected_concepts_file,
    )

    if selected_topics is not None:
        return selected_topics

    return get_course_topics_from_csr(course_df)


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


def run_ace_for_course(
    course: str,
    course_df: pd.DataFrame,
    topics: list[str],
    top_t_percent: float = TOP_T_PERCENT,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Запускает интерактивный ACE-алгоритм для одного курса
    на основе заранее рассчитанного csr_scores.csv.
    """
    g = nx.DiGraph()
    g.add_nodes_from(topics)

    course_df = course_df.sort_values(
        by="prs",
        ascending=False,
    ).reset_index(drop=True)

    limit = math.ceil(len(course_df) * top_t_percent / 100)

    decision_rows = []

    for _, row in course_df.iloc[:limit].iterrows():
        topic_a = row["topic_a"]
        topic_b = row["topic_b"]

        prs = float(row["prs"])
        score_ab = float(row["csr_description_a_to_concept_b"])
        score_ba = float(row["csr_description_b_to_concept_a"])

        # Если между вершинами уже есть путь в любую сторону, пара пропускается, чтобы не нарушать минимальность и не создавать цикл.
        if nx.has_path(g, topic_a, topic_b) or nx.has_path(g, topic_b, topic_a):
            decision_rows.append({
                "course": course,
                "topic_a": topic_a,
                "topic_b": topic_b,
                "prs": prs,
                "csr_description_a_to_concept_b": score_ab,
                "csr_description_b_to_concept_a": score_ba,
                "decision": "skipped_existing_path",
                "source": "",
                "target": "",
            })
            continue

        ans = ask_expert(
            topic_a=topic_a,
            topic_b=topic_b,
            score_ab=score_ab,
            score_ba=score_ba,
            prs=prs,
        )

        if ans == 2:
            decision_rows.append({
                "course": course,
                "topic_a": topic_a,
                "topic_b": topic_b,
                "prs": prs,
                "csr_description_a_to_concept_b": score_ab,
                "csr_description_b_to_concept_a": score_ba,
                "decision": "stop",
                "source": "",
                "target": "",
            })
            break

        if ans == 0:
            source = topic_a
            target = topic_b
            decision = "topic_a_to_topic_b"
        else:
            source = topic_b
            target = topic_a
            decision = "topic_b_to_topic_a"

        g.add_edge(source, target)

        # После добавления ребра оставляем транзитивно сокращённый граф.
        g = nx.transitive_reduction(g)

        decision_rows.append({
            "course": course,
            "topic_a": topic_a,
            "topic_b": topic_b,
            "prs": prs,
            "csr_description_a_to_concept_b": score_ab,
            "csr_description_b_to_concept_a": score_ba,
            "decision": decision,
            "source": source,
            "target": target,
        })

    decisions_df = pd.DataFrame(
        decision_rows,
        columns=[
            "course",
            "topic_a",
            "topic_b",
            "prs",
            "csr_description_a_to_concept_b",
            "csr_description_b_to_concept_a",
            "decision",
            "source",
            "target",
        ],
    )

    return g, decisions_df


def save_graph(
    g: nx.DiGraph,
    course: str,
    decisions_df: pd.DataFrame,
    output_dir: str | Path = GRAPHS_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    """
    Сохраняет граф курса в три файла:
    1. nodes CSV
    2. edges CSV
    3. expert decisions CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    course_name = safe_filename(course)

    nodes_file = output_dir / f"{course_name}_nodes.csv"
    edges_file = output_dir / f"{course_name}_edges.csv"
    decisions_file = output_dir / f"{course_name}_expert_decisions.csv"

    nodes_df = pd.DataFrame([
        {
            "node_id": i,
            "course": course,
            "concept": concept,
        }
        for i, concept in enumerate(g.nodes(), start=1)
    ])

    edges_df = pd.DataFrame([
        {
            "course": course,
            "source": source,
            "target": target,
        }
        for source, target in g.edges()
    ])

    nodes_df.to_csv(nodes_file, index=False, encoding="utf-8-sig")
    edges_df.to_csv(edges_file, index=False, encoding="utf-8-sig")
    decisions_df.to_csv(decisions_file, index=False, encoding="utf-8-sig")

    return nodes_file, edges_file, decisions_file


def print_graph(g: nx.DiGraph, course: str) -> None:
    """
    Печатает итоговые рёбра графа.
    """
    print("\n" + "=" * 80)
    print(f"Итоговые ребра графа для курса: {course}")

    if g.number_of_edges() == 0:
        print("Рёбер нет.")
    else:
        for source, target in g.edges():
            print(f"{source} -> {target}")

    print("=" * 80)

def run_interactive_ace_from_csr(
    csr_input_file: str | Path = CSR_INPUT_FILE,
    selected_concepts_file: str | Path = SELECTED_CONCEPTS_FILE,
    output_dir: str | Path = GRAPHS_OUTPUT_DIR,
    top_t_percent: float = TOP_T_PERCENT,
    courses_to_process: list[str] | None = COURSES_TO_PROCESS,
) -> None:
    """
    Основной сценарий:

    1. Загрузить заранее рассчитанный csr_scores.csv.
    2. Для каждого курса отсортировать пары по PRS.
    3. Запустить интерактивную экспертную разметку.
    4. Построить минимальный граф пререквизитов.
    5. Сохранить nodes, edges и журнал экспертных решений.
    """
    csr_df = load_csr_scores(csr_input_file)

    all_courses = list(csr_df["course"].drop_duplicates())
    courses = filter_courses(
        courses=all_courses,
        courses_to_process=courses_to_process,
    )

    if not courses:
        raise ValueError("Не найдено ни одного курса для запуска ACE.")

    print("\n" + "=" * 80)
    print("Запуск интерактивного ACE-алгоритма")
    print(f"CSR-файл: {Path(csr_input_file).resolve()}")
    print(f"Курсы: {', '.join(courses)}")
    print("=" * 80)

    for course in courses:
        course_df = csr_df[csr_df["course"] == course].copy()

        topics = get_course_topics(
            course=course,
            course_df=course_df,
            selected_concepts_file=selected_concepts_file,
        )

        print("\n" + "#" * 80)
        print(f"Курс: {course}")
        print(f"Количество понятий: {len(topics)}")
        print(f"Количество пар в CSR-файле: {len(course_df)}")
        print("#" * 80)

        graph, decisions_df = run_ace_for_course(
            course=course,
            course_df=course_df,
            topics=topics,
            top_t_percent=top_t_percent,
        )

        nodes_file, edges_file, decisions_file = save_graph(
            g=graph,
            course=course,
            decisions_df=decisions_df,
            output_dir=output_dir,
        )

        print_graph(graph, course=course)

        print(f"Файл вершин: {nodes_file.resolve()}")
        print(f"Файл рёбер: {edges_file.resolve()}")
        print(f"Файл экспертных решений: {decisions_file.resolve()}")


if __name__ == "__main__":
    run_interactive_ace_from_csr()
