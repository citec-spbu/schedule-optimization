import json
from pathlib import Path

INPUT_JSON_PATH = "extracted_data/rpd_2_2.json"
OUTPUT_TX_PATH = "extracted_data/concept_extraction_prompts.txt"

BATCH_SIZE = 5


with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

courses = list(data.items())

output_path = Path(OUTPUT_TX_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)


with open(output_path, "w", encoding="utf-8") as f:
    for prompt_number, start in enumerate(range(0, len(courses), BATCH_SIZE), start=1):
        batch = courses[start:start + BATCH_SIZE]

        course_blocks = []

        for course_title, course_text in batch:
            course_text = str(course_text).strip()

            course_block = f"""
НАЗВАНИЕ КУРСА:
{course_title}

ТЕКСТ КУРСА:
{course_text}
""".strip()

            course_blocks.append(course_block)

        courses_text = "\n\n" + "-" * 80 + "\n\n"
        courses_text = courses_text.join(course_blocks)

        prompt = f"""
ПРОМПТ {prompt_number}

Задача: извлечь учебные понятия из текстов рабочих программ дисциплин.

Нужно проанализировать каждый курс отдельно.

Для каждого курса:
1. Внимательно прочитай название курса.
2. Используй название курса как фильтр релевантности.
3. Проанализируй весь текст курса.
4. Последовательно рассматривай каждое предложение.
5. Внимательно проверяй каждое значимое словосочетание.
6. Выделяй только те понятия, которые явно присутствуют в тексте.
7. Не добавляй понятия от себя.
9. Если в тексте есть полный термин и его фрагмент, возвращай только полный термин.
10. Возвращай полные термины и полные словосочетания, а не обрывки.
11. Не извлекай фамилии, инициалы, имена преподавателей.
12. Не извлекай номера занятий, номера разделов и служебные заголовки.
13. Не извлекай слишком общие слова без предметного смысла.
14. Не извлекай слова: лекция, семинар, практика, лабораторная работа, задача, пример, экзамен, зачет, часы, введение, заключение.

Ответ верни строго в JSON.

Формат ответа:

{{
  "Название курса 1": [
    "понятие 1",
    "понятие 2",
    "понятие 3"
  ],
  "Название курса 2": [
    "понятие 1",
    "понятие 2"
  ]
}}

Тексты для анализа:

{courses_text}
""".strip()

        f.write("=" * 80 + "\n")
        f.write(f"НАЧАЛО ПРОМПТА {prompt_number}\n")
        f.write("=" * 80 + "\n\n")
        f.write(prompt)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"КОНЕЦ ПРОМПТА {prompt_number}\n")
        f.write("=" * 80 + "\n\n\n")


print(f"Готово. Файл сохранен: {OUTPUT_TX_PATH}")
print(f"Всего курсов: {len(courses)}")
print(f"Курсов в одном промпте: {BATCH_SIZE}")
print(f"Всего промптов: {(len(courses) + BATCH_SIZE - 1) // BATCH_SIZE}")
