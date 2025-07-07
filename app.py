import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os
from datetime import datetime
import glob
import zipfile
from bs4 import BeautifulSoup

# Путь к шрифту в контейнере
FONT_PATH = "/fonts/Arial.ttf"

# Загрузка модели
model = YOLO("model.pt")

# Стилизация
TITLE = "Architectural Elements Detector"
DESCRIPTION = """
<h2>Автоматическое определение окон и дверей на архитектурных планах</h2>
<p>Загрузите изображение плана для анализа. Система автоматически обнаружит и пронумерует все окна и двери.</p>
"""
THEME = "soft"
ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"]

# Настройки визуализации
STYLE = {
    "window": {"color": "#1f77b4", "text": "Окно", "emoji": "🪟"},
    "door": {"color": "#2ca02c", "text": "Дверь", "emoji": "🚪"},
    "font_size": 14,
    "border_width": 3,
}


def load_font():
    """Загрузка шрифта с несколькими уровнями fallback"""
    try:
        return ImageFont.truetype(FONT_PATH, STYLE["font_size"])
    except Exception as e:
        print(f"Не удалось загрузить шрифт {FONT_PATH}: {str(e)}")
        try:
            return ImageFont.truetype("arial.ttf", STYLE["font_size"])
        except:
            return ImageFont.load_default()


def detect_objects(image: Image.Image):
    """Основная функция обработки изображения"""
    try:
        img_array = np.array(image)
        results = model(img_array)[0]
        boxes = results.boxes
        names = model.names

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        font = load_font()

        counts = {"window": 0, "door": 0}
        elements_data = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = names[cls]
            conf = float(box.conf[0])

            if label not in STYLE:
                continue

            counts[label] += 1
            element_type = STYLE[label]
            number = counts[label]

            elements_data.append({
                "Тип": element_type["text"],
                "Номер": number,
                "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                "Ширина": x2 - x1, "Высота": y2 - y1,
                "Уверенность": f"{conf:.2f}"
            })

            draw.rectangle([x1, y1, x2, y2], outline=element_type["color"], width=STYLE["border_width"])
            label_text = f"{element_type['emoji']} {element_type['text']} {number}"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1, x1 + text_width + 6, y1 + text_height + 6], fill="white")
            draw.text((x1 + 3, y1 + 3), label_text, fill=element_type["color"], font=font)

        df = pd.DataFrame(elements_data)
        summary_text = f"✅ Найдено: {counts['window']} окон, {counts['door']} дверей"
        table_html = df.to_html(index=False, classes="dataframe",
                                border=0) if not df.empty else "<p>Не обнаружено элементов</p>"

        return img_draw, summary_text, table_html

    except Exception as e:
        error_msg = f"⚠ Ошибка обработки: {str(e)}"
        return image, error_msg, ""


def create_results_zip():
    """Создает ZIP-архив со всеми результатами"""
    try:
        os.makedirs("/app/results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"results_{timestamp}.zip"
        zip_path = f"/app/results/{zip_filename}"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk("/app/results"):
                for file in files:
                    if not file.endswith('.zip'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "/app/results")
                        zipf.write(file_path, arcname)

        return zip_path
    except Exception as e:
        print(f"Error creating zip: {str(e)}")
        return None


def save_results(image: Image.Image, output_image, summary, table):
    """Сохранение результатов анализа с обработкой ошибок"""
    try:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_messages = []

        # Сохраняем изображение
        try:
            img_path = f"results/result_{timestamp}.png"
            if isinstance(output_image, np.ndarray):
                output_image = Image.fromarray(output_image)
            output_image.save(img_path)
            save_messages.append(f"Изображение сохранено: {img_path}")
        except Exception as img_error:
            save_messages.append(f"⚠ Ошибка сохранения изображения: {str(img_error)}")

        # Сохраняем данные
        if table:
            try:
                soup = BeautifulSoup(table, 'html.parser')
                rows = soup.find_all('tr')
                data = []
                headers = [th.get_text() for th in rows[0].find_all('th')]

                for row in rows[1:]:
                    data.append([td.get_text() for td in row.find_all('td')])

                df = pd.DataFrame(data, columns=headers)

                csv_path = f"results/data_{timestamp}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                save_messages.append(f"Данные сохранены (CSV): {csv_path}")

                json_path = f"results/data_{timestamp}.json"
                df.to_json(json_path, orient='records', indent=2, force_ascii=False)
                save_messages.append(f"Данные сохранены (JSON): {json_path}")

            except Exception as data_error:
                try:
                    html_path = f"results/data_{timestamp}.html"
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(table)
                    save_messages.append(f"Данные сохранены (HTML): {html_path}")
                except Exception as html_error:
                    save_messages.append(f"⚠ Ошибка сохранения данных: {str(html_error)}")

        # Создаем ZIP-архив
        zip_path = create_results_zip()
        return "\n".join(save_messages), zip_path

    except Exception as e:
        return f"⛔ Критическая ошибка при сохранении: {str(e)}", None


def download_results():
    """Обработчик скачивания результатов"""
    zip_path = create_results_zip()
    if zip_path and os.path.exists(zip_path):
        return zip_path
    return None


# Кастомный CSS
custom_css = """
.dataframe {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-family: Arial, sans-serif;
}
.dataframe th {
    background-color: #2a2d2e;
    padding: 10px;
    text-align: left;
    font-weight: bold;
    color: #ffffff;
    border-bottom: 2px solid #444;
}
.dataframe td {
    padding: 8px;
    border-bottom: 1px solid #444;
    color: #e0e0e0;
}
.dataframe tr:hover {
    background-color: #3a3d3e;
}
.download-btn {
    background: linear-gradient(45deg, #4CAF50, #2E7D32);
    color: white !important;
    border: none !important;
}
.download-btn:hover {
    background: linear-gradient(45deg, #2E7D32, #1B5E20) !important;
}
"""

# Интерфейс Gradio
with gr.Blocks(title=TITLE, theme=THEME, css=custom_css) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Загрузите план", height=400)
            submit_btn = gr.Button("Анализировать", variant="primary")  # <-- Добавлено

        with gr.Column():
            output_image = gr.Image(type="pil", label="Результат", height=400)
            summary = gr.Textbox(label="Сводка")
            results_table = gr.HTML(label="Детализация")

    with gr.Row():
        save_btn = gr.Button("💾 Сохранить результаты", variant="primary")
        download_btn = gr.Button("📥 Скачать все результаты", elem_classes="download-btn")

    save_output = gr.Textbox(label="Статус сохранения")
    download_output = gr.File(label="Архив результатов", visible=True)

    # Обработчики событий (теперь эти переменные определены)
    submit_btn.click(
        fn=detect_objects,
        inputs=input_image,
        outputs=[output_image, summary, results_table]
    )

    save_btn.click(
        fn=save_results,
        inputs=[input_image, output_image, summary, results_table],
        outputs=[save_output, download_output]
    )

    download_btn.click(
        fn=download_results,
        outputs=download_output
    )

    # Примеры для быстрого тестирования
    gr.Examples(
        examples=["example1.jpg", "example2.png"],
        inputs=input_image,
        label="Примеры планов (нажмите чтобы загрузить)"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="favicon.ico.png",
        share=False
    )
