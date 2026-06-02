import json
import os
import time
import uuid
from pathlib import Path

import requests
from openai import OpenAI


# =========================
# 1. НАСТРОЙКИ
# =========================

AUTH_KEY_ENV_NAME = "GIGACHAT_AUTH_KEY"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "extracted_data"

INPUT_CONCEPTS_PATH = DATA_DIR / "selected_concepts.json"
OUTPUT_DESCRIPTIONS_PATH = DATA_DIR / "courses_topics.json"
ERRORS_JSON_PATH = DATA_DIR / "courses_topics_errors.json"

MODEL_NAME = "GigaChat-2-Pro"
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

# None = обработать все курсы из INPUT_CONCEPTS_PATH.
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

def load_concepts(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект формата {курс: [понятия]}.")

    result: dict[str, list[str]] = {}

    for course_title, concepts in data.items():
        if not isinstance(course_title, str):
            continue

        if not isinstance(concepts, list):
            continue

        cleaned_concepts = []

        for concept in concepts:
            if not isinstance(concept, str):
                continue

            concept = concept.strip()

            if concept:
                cleaned_concepts.append(concept)

        course_title = course_title.strip()

        if course_title and cleaned_concepts:
            result[course_title] = cleaned_concepts

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
# 6. ПОДГОТОВКА КУРСОВ
# =========================

def filter_courses(
    courses: dict[str, list[str]],
    courses_to_process: list[str] | None = None,
) -> dict[str, list[str]]:
    if courses_to_process is None:
        return courses

    return {
        course: courses[course]
        for course in courses_to_process
        if course in courses
    }


# =========================
# 7. PROMPTS
# =========================

SYSTEM_PROMPT = """
Ты пишешь учебные описания понятий в рамках университетского курса.

Требования:
- объясняй понятие строго в контексте названия курса;
- не уходи в посторонние темы;
- описание должно быть достаточно содержательным;
- объем описания: примерно 10 предложений;
- стиль: учебный, академичный, ясный;
- не используй markdown;
- не используй списки;
- не добавляй вступлений вроде "в рамках данного курса";
- возвращай только сам текст описания.
""".strip()


def build_description_prompt(course_title: str, concept: str) -> str:
    return f"""
Название курса:
{course_title}

Понятие:
{concept}

Напиши связное учебное описание этого понятия в рамках данного курса объемом примерно 10 предложений.
Описание должно объяснять смысл понятия, его роль в курсе и связь с близкими учебными темами.
Не добавляй внешние термины, если они не нужны для объяснения.

Верни только готовый текст описания без заголовков, markdown-разметки и дополнительных комментариев.
""".strip()


# =========================
# 8. ЗАПРОС К МОДЕЛИ
# =========================

def generate_concept_description(
    client: OpenAI,
    course_title: str,
    concept: str,
) -> str:
    user_prompt = build_description_prompt(course_title, concept)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.3,
            )

            description = response.choices[0].message.content.strip()

            if not description:
                raise ValueError("Модель вернула пустое описание.")

            return description

        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_DELAY_SECONDS * attempt)
                continue

            raise RuntimeError(
                f"Не удалось получить описание после {MAX_RETRIES} попыток."
            ) from last_error

    raise RuntimeError("Не удалось получить описание.")


# =========================
# 9. ОБРАБОТКА ВСЕХ ПОНЯТИЙ
# =========================

def process_all_concepts(
    client: OpenAI,
    concepts_by_course: dict[str, list[str]],
    output_path: str | Path,
    errors_path: str | Path,
) -> None:
    descriptions = load_existing_json(output_path)
    errors = load_existing_json(errors_path)

    total_courses = len(concepts_by_course)

    for course_index, (course_title, concepts) in enumerate(
        concepts_by_course.items(),
        start=1,
    ):
        if course_title not in descriptions or not isinstance(descriptions[course_title], dict):
            descriptions[course_title] = {}

        if course_title not in errors or not isinstance(errors.get(course_title), dict):
            errors[course_title] = {}

        print(f"[{course_index}/{total_courses}] Курс: {course_title}")

        for concept_index, concept in enumerate(concepts, start=1):
            if concept in descriptions[course_title]:
                print(f"  [{concept_index}/{len(concepts)}] Уже есть описание: {concept}")
                continue

            print(f"  [{concept_index}/{len(concepts)}] Генерация описания: {concept}")

            try:
                description = generate_concept_description(
                    client=client,
                    course_title=course_title,
                    concept=concept,
                )

                descriptions[course_title][concept] = description

                if concept in errors[course_title]:
                    del errors[course_title][concept]

                save_json(output_path, descriptions)
                save_json(errors_path, errors)

                print("    Готово.")

            except Exception as error:
                errors[course_title][concept] = str(error)

                save_json(output_path, descriptions)
                save_json(errors_path, errors)

                print(f"    Ошибка: {error}")

            time.sleep(REQUEST_DELAY_SECONDS)

        if not errors.get(course_title):
            errors.pop(course_title, None)
            save_json(errors_path, errors)


# =========================
# 10. ОСНОВНОЙ ЗАПУСК
# =========================

def main() -> None:
    auth_key = get_auth_key_from_env()
    client = create_client(auth_key)

    concepts_by_course = load_concepts(INPUT_CONCEPTS_PATH)
    concepts_by_course = filter_courses(concepts_by_course, COURSES_TO_PROCESS)

    process_all_concepts(
        client=client,
        concepts_by_course=concepts_by_course,
        output_path=OUTPUT_DESCRIPTIONS_PATH,
        errors_path=ERRORS_JSON_PATH,
    )


if __name__ == "__main__":
    main()
