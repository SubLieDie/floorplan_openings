import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Загрузка модели
model = YOLO("model.pt")


def detect_objects(image: Image.Image):
    img_array = np.array(image)
    results = model(img_array)[0]

    boxes = results.boxes
    names = model.names

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    allowed_labels = ['window', 'door']  # ← Подставь точные названия

    window_count = 0
    door_count = 0
    coords_text = ''

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = names[cls]

        if label not in allowed_labels:
            continue

        if label == 'window':
            window_count += 1
            number = window_count
            color = "blue"
            label_text = f"Окно {number}"
            coords_text += f"Окно {number}: ({x1}, {y1}), ({x2}, {y2})\n"

        elif label == 'door':
            door_count += 1
            number = door_count
            color = "green"
            label_text = f"Дверь {number}"
            coords_text += f"Дверь {number}: ({x1}, {y1}), ({x2}, {y2})\n"

        # Рисуем прямоугольник и подпись
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 3, y1 + 3), label_text, fill=color)

    return img_draw, coords_text.strip()


# Интерфейс Gradio
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Загрузите архитектурный план"),
    outputs=[
        gr.Image(type="pil", label="Обработанное изображение"),
        gr.Textbox(label="Координаты окон и проемов", lines=10)
    ],
    title="Детекция окон и дверей на плане"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
