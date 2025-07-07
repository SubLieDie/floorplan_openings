FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxml2 \
    libxslt1.1 \
    fonts-freefont-ttf \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей для lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    && pip install --no-cache-dir lxml>=4.9.0 \
    && apt-get remove -y gcc python3-dev libxml2-dev libxslt1-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем сначала requirements.txt для лучшего кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install beautifulsoup4 html5lib

# Копируем шрифт Arial
COPY fonts/Arial.ttf /usr/share/fonts/truetype/arial/

# Обновляем кэш шрифтов
RUN fc-cache -f -v

# Создаем папку для результатов
RUN mkdir -p /app/results && chmod -R 777 /app/results

# Копируем остальные файлы
COPY . .

CMD ["python", "app.py"]