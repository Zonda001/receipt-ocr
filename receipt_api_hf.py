"""
receipt_api_hf.py - FastAPI для Hugging Face Spaces

Оптимізовано для HF:
- Порт 7860
- Lazy loading моделей
- Кешування результатів
- Мінімальне використання пам'яті

URL: https://YOUR_USERNAME-receipt-ocr.hf.space
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile
import os
from pathlib import Path
import hashlib
from datetime import datetime
import json
import traceback
from typing import Optional
import base64

# Lazy import для економії пам'яті
ocr_engine: Optional[object] = None
model_load_time: Optional[datetime] = None

app = FastAPI(
    title="Receipt OCR API 🇺🇦",
    description="API для розпізнавання українських чеків",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS для мобільних додатків
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Простий кеш
CACHE = {}
CACHE_MAX = 100


def get_ocr_engine():
    """Завантажити OCR при першому запиті (lazy loading)"""
    global ocr_engine, model_load_time

    if ocr_engine is None:
        print("🔄 Завантаження EasyOCR моделей...")
        start = datetime.now()

        from receipt_ocr_ultimate import UltimateReceiptOCR

        # Тільки українська для економії пам'яті
        ocr_engine = UltimateReceiptOCR(languages=['uk', 'en'])

        model_load_time = datetime.now()
        load_duration = (model_load_time - start).total_seconds()
        print(f"✅ Моделі завантажені за {load_duration:.1f}s")

    return ocr_engine


@app.on_event("startup")
async def startup_event():
    """Старт без завантаження моделей (lazy load)"""
    print("🚀 Receipt OCR API запущено на Hugging Face Spaces")
    print("📡 Порт: 7860")
    print("⏳ Моделі завантажаться при першому запиті...")


@app.get("/", response_class=HTMLResponse)
def root():
    """Головна сторінка з інтерфейсом"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Receipt OCR API 🧾</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h1 { font-size: 2.5em; margin-bottom: 10px; }
            .emoji { font-size: 3em; }
            .upload-area {
                border: 3px dashed rgba(255,255,255,0.5);
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin: 30px 0;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                border-color: white;
                background: rgba(255,255,255,0.1);
            }
            button {
                background: white;
                color: #667eea;
                border: none;
                padding: 15px 40px;
                font-size: 16px;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover { transform: scale(1.05); }
            #result {
                margin-top: 30px;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                display: none;
            }
            .item { 
                padding: 10px;
                margin: 5px 0;
                background: rgba(255,255,255,0.1);
                border-radius: 5px;
            }
            a { color: white; text-decoration: underline; }
            .endpoint {
                background: rgba(0,0,0,0.2);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="emoji">🧾🇺🇦</div>
            <h1>Receipt OCR API</h1>
            <p>API для розпізнавання українських чеків</p>

            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📸 Завантажте фото чеку</h3>
                <p>JPG або PNG, до 10MB</p>
                <input type="file" id="fileInput" accept="image/*" style="display:none">
            </div>

            <button onclick="uploadReceipt()">Розпізнати чек</button>

            <div id="result"></div>

            <hr style="margin: 40px 0; opacity: 0.3;">

            <h3>📡 API Endpoints</h3>
            <div class="endpoint">
                <strong>POST</strong> /api/ocr<br>
                Розпізнати чек (multipart/form-data)
            </div>
            <div class="endpoint">
                <strong>POST</strong> /api/ocr/base64<br>
                Розпізнати чек (JSON base64)
            </div>
            <div class="endpoint">
                <strong>GET</strong> /health<br>
                Перевірка здоров'я API
            </div>

            <p style="margin-top: 30px;">
                📚 <a href="/docs">API Documentation</a> | 
                <a href="/redoc">ReDoc</a>
            </p>
        </div>

        <script>
            let selectedFile = null;

            document.getElementById('fileInput').addEventListener('change', (e) => {
                selectedFile = e.target.files[0];
                if (selectedFile) {
                    document.querySelector('.upload-area h3').textContent = 
                        '✓ ' + selectedFile.name;
                }
            });

            async function uploadReceipt() {
                if (!selectedFile) {
                    alert('Виберіть файл!');
                    return;
                }

                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>⏳ Обробка...</p>';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const response = await fetch('/api/ocr', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.success) {
                        let html = '<h3>✅ Результат</h3>';

                        if (data.receipt.items && data.receipt.items.length > 0) {
                            html += '<h4>Товари:</h4>';
                            data.receipt.items.forEach(item => {
                                html += `<div class="item">
                                    ${item.name} - ${item.price} грн
                                </div>`;
                            });
                        }

                        html += '<h4>Підсумки:</h4>';
                        if (data.receipt.suma) html += `<div class="item">Сума: ${data.receipt.suma} грн</div>`;
                        if (data.receipt.pdv) html += `<div class="item">ПДВ: ${data.receipt.pdv} грн</div>`;
                        if (data.receipt.do_splaty) html += `<div class="item">До сплати: ${data.receipt.do_splaty} грн</div>`;

                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = '<p>❌ ' + (data.error || 'Помилка обробки') + '</p>';
                    }
                } catch (err) {
                    resultDiv.innerHTML = '<p>❌ Помилка: ' + err.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return html


@app.get("/health")
def health_check():
    """Перевірка здоров'я API"""
    return {
        "status": "healthy",
        "model_loaded": ocr_engine is not None,
        "model_load_time": model_load_time.isoformat() if model_load_time else None,
        "cache_size": len(CACHE)
    }


@app.post("/api/ocr")
@app.post("/api/ocr/")
async def ocr_from_file(
        file: UploadFile = File(...),
        use_cache: bool = True
):
    """
    Розпізнати чек з файлу

    Args:
        file: Зображення чеку (JPG, PNG)
        use_cache: Використовувати кеш результатів

    Returns:
        JSON з розпізнаними даними
    """

    # Валідація типу
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(400, "Підтримуються тільки JPG/PNG файли")

    # Читаємо файл
    content = await file.read()

    # Перевірка розміру
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(400, "Файл завеликий (максимум 10MB)")

    # Кеш
    file_hash = hashlib.md5(content).hexdigest()

    if use_cache and file_hash in CACHE:
        print(f"🎯 Cache hit: {file_hash[:8]}")
        return {**CACHE[file_hash], "from_cache": True}

    # Обробка
    tmp_path = None
    try:
        # Завантажити OCR (lazy)
        engine = get_ocr_engine()

        # Зберегти тимчасово
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        print(f"📸 Обробка: {file.filename} ({len(content)} bytes)")

        # OCR
        ocr_data = engine.process_image(tmp_path)

        if not ocr_data:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "Текст не знайдено на зображенні",
                    "receipt": None
                }
            )

        # Структуризувати
        receipt = engine.structure_receipt(ocr_data)

        result = {
            "success": True,
            "receipt": receipt,
            "meta": {
                "filename": file.filename,
                "text_regions_found": len(ocr_data),
                "items_found": len(receipt.get('items', []))
            }
        }

        # Зберегти в кеш
        if use_cache:
            if len(CACHE) >= CACHE_MAX:
                CACHE.pop(next(iter(CACHE)))  # Видалити найстаріший
            CACHE[file_hash] = result

        return {**result, "from_cache": False}

    except Exception as e:
        print(f"❌ Помилка: {traceback.format_exc()}")
        raise HTTPException(500, f"Помилка обробки: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr/base64")
@app.post("/api/ocr/base64/")
async def ocr_from_base64(data: dict):
    """
    Розпізнати чек з base64 (для мобільних додатків)

    Body:
    {
        "image": "base64_string",
        "use_cache": true
    }
    """

    if "image" not in data:
        raise HTTPException(400, "Поле 'image' обов'язкове")

    # Декодувати
    try:
        image_data = base64.b64decode(data["image"])
    except:
        raise HTTPException(400, "Невалідний base64")

    # Кеш
    file_hash = hashlib.md5(image_data).hexdigest()
    use_cache = data.get("use_cache", True)

    if use_cache and file_hash in CACHE:
        return {**CACHE[file_hash], "from_cache": True}

    # Обробка
    tmp_path = None
    try:
        engine = get_ocr_engine()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        print(f"📸 Обробка base64 ({len(image_data)} bytes)")

        ocr_data = engine.process_image(tmp_path)

        if not ocr_data:
            return {
                "success": False,
                "error": "Текст не знайдено",
                "receipt": None
            }

        receipt = engine.structure_receipt(ocr_data)

        result = {
            "success": True,
            "receipt": receipt,
            "meta": {
                "text_regions_found": len(ocr_data),
                "items_found": len(receipt.get('items', []))
            }
        }

        if use_cache:
            if len(CACHE) >= CACHE_MAX:
                CACHE.pop(next(iter(CACHE)))
            CACHE[file_hash] = result

        return {**result, "from_cache": False}

    except Exception as e:
        print(f"❌ Помилка: {traceback.format_exc()}")
        raise HTTPException(500, f"Помилка: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/stats")
def get_stats():
    """Статистика API"""
    return {
        "model_loaded": ocr_engine is not None,
        "model_load_time": model_load_time.isoformat() if model_load_time else None,
        "cache_entries": len(CACHE),
        "cache_max": CACHE_MAX
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)