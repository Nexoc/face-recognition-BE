# Используем официальный базовый образ Python 3.10
FROM python:3.10

# Устанавливаем системные зависимости, необходимые для face_recognition и dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей (тут у тебя .md, лучше переименовать в .txt)
COPY requirements.txt .

# Устанавливаем зависимости Python из файла
RUN pip install --no-cache-dir -r requirements.txt

# Копируем папку app с Python-кодом внутрь контейнера
COPY ./app ./app

# Запускаем FastAPI с включённым hot-reload (для разработки)
# --reload включает автоматическую перезагрузку FastAPI при любом изменении .py файлов.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
