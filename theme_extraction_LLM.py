import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


# =========================
# 1. НАСТРОЙКИ
# =========================

AUTH_KEY_ENV_NAME = "GIGACHAT_AUTH_KEY"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "extracted_data"

INPUT_JSON_PATH = DATA_DIR / "rpd_2_2.json"
OUTPUT_JSON_PATH = DATA_DIR / "educational_concepts.json"
ERRORS_JSON_PATH = DATA_DIR / "educational_concepts_errors.json"

MODEL_NAME = "GigaChat-2-Pro"
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

# None = обработать все курсы из INPUT_JSON_PATH.
COURSES_TO_PROCESS: list[str] | None = None


# =========================
# 2. ЧТЕНИЕ AUTH KEY ИЗ ОКРУЖЕНИЯ
# =========================

def get_auth_key_from_env(env_name: str = AUTH_KEY_ENV_NAME) -> str:
    auth_key = os.getenv(env_name)

    if not auth_key:
        raise RuntimeError(
            f"Не найдена переменная окружения {env_name}. "
            f"Добавьте в окружение ключ авторизации GigaChat."
        )

    auth_key = auth_key.strip()

    if not auth_key:
        raise RuntimeError(
            f"Переменная окружения {env_name} задана, но она пустая."
        )

    return auth_key


# =========================
# 3. ПОЛУЧЕНИЕ ACCESS TOKEN
# =========================

def get_access_token(auth_key: str) -> str:
    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Bearer {auth_key}",
    }

    data = {
        "scope": "GIGACHAT_API_PERS",
    }

    response = requests.post(token_url, headers=headers, data=data, timeout=30)
    response.raise_for_status()

    return response.json()["access_token"]


# =========================
# 4. СОЗДАНИЕ КЛИЕНТА
# =========================

def create_client(auth_key: str) -> OpenAI:
    access_token = get_access_token(auth_key)

    return OpenAI(
        api_key=access_token,
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
    )


# =========================
# 5. ЧТЕНИЕ И СОХРАНЕНИЕ JSON
# =========================

def load_courses(path: str | Path) -> dict[str, str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("JSON-файл пуст.")

    if not isinstance(data, dict):
        raise TypeError(
            "Ожидался JSON-объект формата: {название_курса: текст_курса}."
        )

    result: dict[str, str] = {}

    for course_title, course_text in data.items():
        if not isinstance(course_title, str):
            continue

        if not isinstance(course_text, str):
            course_text = str(course_text)

        course_title = course_title.strip()
        course_text = course_text.strip()

        if course_title and course_text:
            result[course_title] = course_text

    return result


def load_existing_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 6. ПОДГОТОВКА ТЕКСТА
# =========================

def filter_courses(
    courses: dict[str, str],
    courses_to_process: list[str] | None = None,
) -> dict[str, str]:
    if courses_to_process is None:
        return courses

    return {
        course: courses[course]
        for course in courses_to_process
        if course in courses
    }


def get_nonempty_lines(source_text: str) -> list[str]:
    return [
        line.strip()
        for line in source_text.splitlines()
        if line.strip()
    ]


# =========================
# 7. PROMPTS
# =========================

SYSTEM_PROMPT = """
Ты извлекаешь из строк текста понятия, относящиеся к теме курса.

Требования к анализу:
- анализируй каждую непустую строку отдельно;
- учитывай название курса как фильтр релевантности;
- извлекай только те понятия, которые явно присутствуют в строке или однозначно читаются с учетом артефактов извлечения текста;
- не добавляй ничего от себя;
- если в строке нет подходящих понятий, возвращай пустой список;

Требования к формату ответа:
- верни только один JSON-объект;
- ответ должен начинаться символом { и заканчиваться символом };
- верхний ключ только один: "lines";
- каждый объект списка обязан содержать ключи "line_number" и "concepts";
- значение "concepts" всегда должно быть списком строк, даже если список пустой;
- не добавляй никаких пояснений вне JSON.

Пример допустимого формата:
{
  "lines": [
    {
      "line_number": 1,
      "concepts": ["понятие 1", "понятие 2"]
    },
    {
      "line_number": 2,
      "concepts": []
    }
  ]
}
""".strip()


def build_user_prompt(course_title: str, lines: list[str]) -> str:
    input_lines = [
        {
            "line_number": i,
            "text": line,
        }
        for i, line in enumerate(lines, start=1)
    ]

    return f"""
Название курса:
{course_title}

Ниже дан список непустых строк текста.

Задача:
для каждой строки выделить понятия, которые:
1) явно присутствуют в тексте строки;
2) относятся к теме данного курса.

Дополнительные указания:
- нельзя придумывать понятия, которых нет в строке;
- нельзя пропускать строки;
- не извлекай неполные фрагменты терминов;
- не извлекай фамилии, инициалы и служебные заголовки занятий;
- если в строке есть полный термин и его фрагмент, верни только полный термин;
- возвращай полный термин или полное словосочетание, а не его обрывок;
- не возвращай неполные фрагменты, одиночные служебные слова, слишком общие одиночные слова, фамилии и инициалы;
- не возвращай как понятия служебные слова и заголовки: лекция, лекции, практическое, практические, лабораторная, лабораторные, семинар, семинары, пример, примеры, задача, задачи, экзамен, зачет, введение, заключение, часы.

Верни строго один JSON-объект без какого-либо дополнительного текста.

Строки:
{json.dumps(input_lines, ensure_ascii=False, indent=2)}
""".strip()


# =========================
# 8. ЗАПРОС К МОДЕЛИ
# =========================

def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"В ответе модели не найден JSON-объект: {raw_text[:500]}")

    json_text = text[start:end + 1]
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("Ответ модели должен быть JSON-объектом.")

    return parsed


def extract_concepts_for_course(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0,
            )

            raw_answer = response.choices[0].message.content.strip()
            return extract_json_object(raw_answer)

        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY_SECONDS * attempt)
                continue

            raise RuntimeError(
                f"Не удалось получить корректный ответ модели после {MAX_RETRIES} попыток."
            ) from last_error

    raise RuntimeError("Не удалось получить ответ модели.")


# =========================
# 9. НОРМАЛИЗАЦИЯ И ОБЪЕДИНЕНИЕ ПОНЯТИЙ
# =========================

def normalize_concept(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r,;:.!?\"'«»()[]{}")

    return text


def collect_concepts(parsed_response: dict[str, Any]) -> list[str]:
    result = []
    seen = set()

    lines = parsed_response.get("lines", [])

    if not isinstance(lines, list):
        raise ValueError('В ответе модели ключ "lines" должен быть списком.')

    for item in lines:
        if not isinstance(item, dict):
            continue

        concepts = item.get("concepts", [])

        if not isinstance(concepts, list):
            continue

        for concept in concepts:
            if not isinstance(concept, str):
                continue

            concept = normalize_concept(concept)

            if not concept:
                continue

            if concept not in seen:
                seen.add(concept)
                result.append(concept)

    return result


# =========================
# 10. ОБРАБОТКА ВСЕХ КУРСОВ
# =========================

def process_all_courses(
    client: OpenAI,
    courses: dict[str, str],
    output_path: str | Path,
    errors_path: str | Path,
) -> None:
    results = load_existing_json(output_path)
    errors = load_existing_json(errors_path)

    total_courses = len(courses)

    for course_index, (course_title, source_text) in enumerate(courses.items(), start=1):
        if course_title in results:
            print(f"[{course_index}/{total_courses}] Уже обработан: {course_title}")
            continue

        print(f"[{course_index}/{total_courses}] Обработка курса: {course_title}")

        try:
            lines = get_nonempty_lines(source_text)
            user_prompt = build_user_prompt(course_title, lines)

            parsed = extract_concepts_for_course(
                client=client,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            concepts = collect_concepts(parsed)
            results[course_title] = concepts

            if course_title in errors:
                del errors[course_title]

            save_json(output_path, results)
            save_json(errors_path, errors)

            print(f"  Готово. Извлечено понятий: {len(concepts)}")

        except Exception as error:
            errors[course_title] = str(error)

            save_json(output_path, results)
            save_json(errors_path, errors)

            print(f"  Ошибка: {error}")

        time.sleep(REQUEST_DELAY_SECONDS)


# =========================
# 11. ОСНОВНОЙ ЗАПУСК
# =========================

def main() -> None:
    auth_key = get_auth_key_from_env()
    client = create_client(auth_key)

    courses = load_courses(INPUT_JSON_PATH)
    courses = filter_courses(courses, COURSES_TO_PROCESS)

    process_all_courses(
        client=client,
        courses=courses,
        output_path=OUTPUT_JSON_PATH,
        errors_path=ERRORS_JSON_PATH,
    )


if __name__ == "__main__":
    main()
