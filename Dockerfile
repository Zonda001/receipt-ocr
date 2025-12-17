# Dockerfile для Hugging Face Spaces
# Оптимізований для швидкого завантаження та економії пам'яті

FROM python:3.10-slim

# Встановити робочу директорію
WORKDIR /app

# Встановити системні залежності для OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Копіювати requirements і встановити залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювати код
COPY receipt_ocr_ultimate.py .
COPY receipt_api_hf.py .

# Створити директорію для тимчасових файлів
RUN mkdir -p /tmp/ocr_cache

# Експортувати порт 7860 (стандарт для HF Spaces)
EXPOSE 7860

# Встановити змінні середовища
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Запустити FastAPI на порту 7860
CMD ["uvicorn", "receipt_api_hf:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]