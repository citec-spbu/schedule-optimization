import json
from pathlib import Path
from typing import Dict, List


# =========================
# НАСТРОЙКИ
# =========================

INPUT_JSON_NAME = "extracted_data/selected_concepts.json"
OUTPUT_TXT_NAME = "extracted_data/concept_description_prompts.txt"

SENTENCES_PER_TOPIC = 10
MAX_OUTPUT_SENTENCES_PER_PROMPT = 80
MAX_TOPICS_PER_PROMPT = MAX_OUTPUT_SENTENCES_PER_PROMPT // SENTENCES_PER_TOPIC


# =========================
# ЧТЕНИЕ JSON
# =========================

def get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_input_json(script_dir: Path) -> Path:
    if INPUT_JSON_NAME:
        json_path = script_dir / INPUT_JSON_NAME

        if not json_path.exists():
            raise FileNotFoundError(f"Файл не найден: {json_path}")

        return json_path

    json_files = list(script_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("В папке со скриптом не найден .json файл.")

    if len(json_files) > 1:
        names = "\n".join(path.name for path in json_files)
        raise RuntimeError(
            "В папке найдено несколько .json файлов. "
            "Укажи нужный файл в INPUT_JSON_NAME.\n\n"
            f"Найденные файлы:\n{names}"
        )

    return json_files[0]


def load_courses(json_path: Path) -> Dict[str, List[str]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON должен иметь структуру: {название_курса: список_тем}")

    courses = {}

    for course_name, topics in data.items():
        if not isinstance(topics, list):
            continue

        cleaned_topics = [
            topic.strip()
            for topic in topics
            if isinstance(topic, str) and topic.strip()
        ]

        if cleaned_topics:
            courses[course_name.strip()] = cleaned_topics

    return courses


# =========================
# ГЕНЕРАЦИЯ ПРОМПТОВ
# =========================

def split_topics(topics: List[str], chunk_size: int) -> List[List[str]]:
    return [
        topics[i:i + chunk_size]
        for i in range(0, len(topics), chunk_size)
    ]


def build_prompt(course_name: str, topics: List[str]) -> str:
    topics_text = "\n".join(
        f"- {topic}"
        for topic in topics
    )

    prompt = f"""Задача: подготовить учебные описания тем в рамках конкретного курса.

Название курса:
{course_name}

Темы:
{topics_text}

Требования:
1. Подготовь описание для каждой темы из списка.
2. Каждое описание должно быть примерно на 10 предложений.
3. Описание должно раскрывать тему именно в контексте курса "{course_name}".
4. Пиши академично, но понятно.
5. В описании объясняй смысл темы, говори, что означает данное понятие

Верни ответ строго в формате JSON.

Структура ответа:

{{
  "{course_name}": {{
    "точное название первой темы": "описание первой темы примерно на 10 предложений",
    "точное название второй темы": "описание второй темы примерно на 10 предложений"
  }}
}}
"""
    return prompt.strip()


def generate_prompts(courses: Dict[str, List[str]]) -> List[str]:
    prompts = []

    for course_name, topics in courses.items():
        topic_chunks = split_topics(topics, MAX_TOPICS_PER_PROMPT)

        for chunk in topic_chunks:
            prompt = build_prompt(course_name, chunk)
            prompts.append(prompt)

    return prompts


def save_prompts(prompts: List[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for index, prompt in enumerate(prompts, start=1):
            file.write(f"==================== PROMPT {index} ====================\n")
            file.write(prompt)
            file.write("\n\n")


# =========================
# ЗАПУСК
# =========================

def main() -> None:
    script_dir = get_script_dir()

    input_json_path = find_input_json(script_dir)
    output_txt_path = script_dir / OUTPUT_TXT_NAME

    courses = load_courses(input_json_path)
    prompts = generate_prompts(courses)

    save_prompts(prompts, output_txt_path)

    total_topics = sum(len(topics) for topics in courses.values())

    print(f"Входной файл: {input_json_path.name}")
    print(f"Курсов обработано: {len(courses)}")
    print(f"Тем обработано: {total_topics}")
    print(f"Тем в одном промпте: {MAX_TOPICS_PER_PROMPT}")
    print(f"Промптов создано: {len(prompts)}")
    print(f"Результат сохранён в файл: {output_txt_path.name}")


if __name__ == "__main__":
    main()
