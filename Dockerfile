FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY api_production.py .

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "api_production:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:$PORT", "--timeout", "120"]