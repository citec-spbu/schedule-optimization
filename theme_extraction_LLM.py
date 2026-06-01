import json
import uuid
import re
import os
import time
import requests
from openai import OpenAI


# =========================
# 1. НАСТРОЙКИ
# =========================
AUTH_KEY_ENV_NAME = "GIGACHAT_AUTH_KEY"

INPUT_JSON_PATH = "RPD_SPBU\\rpd_2_2.json"
OUTPUT_JSON_PATH = "topics_all_courses.json"
ERRORS_JSON_PATH = "topics_all_courses_errors.json"

MODEL_NAME = "GigaChat-2-Pro"
REQUEST_DELAY_SECONDS = 0.5


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
        "scope": "GIGACHAT_API_PERS"
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
        base_url="https://gigachat.devices.sberbank.ru/api/v1"
    )


# =========================
# 5. ЗАГРУЗКА ВСЕХ КУРСОВ ИЗ JSON
# =========================
def load_courses(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("JSON-файл пуст.")

    if not isinstance(data, dict):
        raise TypeError("Ожидался JSON-объект формата: {название_курса: текст_курса}.")

    return data


# =========================
# 6. ЗАГРУЗКА УЖЕ СОХРАНЕННЫХ РЕЗУЛЬТАТОВ
# =========================
def load_existing_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 7. ПОДГОТОВКА НЕПУСТЫХ СТРОК
# =========================
def get_nonempty_lines(source_text: str) -> list[str]:
    return [
        line.strip()
        for line in source_text.splitlines()
        if line.strip()
    ]


# =========================
# 8. SYSTEM PROMPT
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


# =========================
# 9. USER PROMPT ДЛЯ ОДНОГО КУРСА
# =========================
def build_user_prompt(course_title: str, lines: list[str]) -> str:
    input_lines = [
        {
            "line_number": i,
            "text": line
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
# 10. ЗАПРОС К МОДЕЛИ
# =========================
def extract_concepts_for_course(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str
) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    raw_answer = response.choices[0].message.content.strip()

    return json.loads(raw_answer)


# =========================
# 11. НОРМАЛИЗАЦИЯ ПОНЯТИЙ
# =========================
def normalize_concept(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r,;:.!?\"'«»()[]{}")

    return text


# =========================
# 12. ОБЪЕДИНЕНИЕ ПОНЯТИЙ
# =========================
def collect_concepts(parsed_response: dict) -> list[str]:
    result = []
    seen = set()

    for item in parsed_response.get("lines", []):
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
# 13. СОХРАНЕНИЕ JSON
# =========================
def save_json(output_path: str, data: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 14. ОБРАБОТКА ВСЕХ КУРСОВ
# =========================
def process_all_courses(
    client: OpenAI,
    courses: dict[str, str],
    output_path: str,
    errors_path: str
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
                user_prompt=user_prompt
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
# 15. ОСНОВНОЙ ЗАПУСК
# =========================
def main() -> None:
    auth_key = get_auth_key_from_env()
    client = create_client(auth_key)

    courses = load_courses(INPUT_JSON_PATH)

    process_all_courses(
        client=client,
        courses=courses,
        output_path=OUTPUT_JSON_PATH,
        errors_path=ERRORS_JSON_PATH
    )


if __name__ == "__main__":
    main()
