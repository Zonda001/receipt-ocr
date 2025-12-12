"""
receipt_api.py - Production REST API для розпізнавання чеків

ВАЖЛИВО: Оптимізовано для Render.com Free tier
- Швидкий старт (порт відкривається негайно)
- Async завантаження EasyOCR моделей
- Graceful handling холодного старту

Деплой на Render.com:
    1. Завантажте файли на GitHub
    2. Підключіть репозиторій до Render
    3. Render автоматично задеплоїть

Локальний запуск:
    pip install -r requirements.txt
    uvicorn receipt_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import io
import re
import os
import logging
import asyncio
from datetime import datetime

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Receipt OCR API",
    description="API для розпізнавання українських чеків",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальні змінні
reader = None
reader_loading = False
reader_load_error = None
server_start_time = datetime.now()


async def load_easyocr_async():
    """Асинхронне завантаження EasyOCR у фоновому режимі"""
    global reader, reader_loading, reader_load_error

    try:
        reader_loading = True
        logger.info("🔄 Початок завантаження EasyOCR моделей...")

        # Імпорт тут щоб не блокувати старт
        import easyocr

        # Завантажуємо у sync коді (EasyOCR не async)
        loop = asyncio.get_event_loop()
        reader = await loop.run_in_executor(
            None,
            lambda: easyocr.Reader(
                ['uk', 'en', 'ru'],
                gpu=False,
                verbose=False,
                download_enabled=True,
                quantize=True  # Використовуємо quantized моделі (швидше, менше RAM)
            )
        )

        logger.info("✅ EasyOCR завантажено успішно!")
        reader_loading = False
        reader_load_error = None

    except Exception as e:
        logger.error(f"❌ Помилка завантаження EasyOCR: {e}")
        reader_loading = False
        reader_load_error = str(e)
        reader = None


@app.on_event("startup")
async def startup_event():
    """Швидкий старт сервера без блокування"""
    logger.info("🚀 Сервер стартує...")
    logger.info("📡 Порт відкривається...")

    # Запускаємо завантаження моделей У ФОНІ
    asyncio.create_task(load_easyocr_async())

    logger.info("✅ Сервер готовий! (OCR завантажується у фоні)")


# ======================== MODELS ========================

class ReceiptItem(BaseModel):
    name: str
    price: str
    confidence: float


class ReceiptResponse(BaseModel):
    success: bool
    items: List[ReceiptItem]
    suma: Optional[str] = None
    pdv: Optional[str] = None
    doSplaty: Optional[str] = None
    discount: Optional[str] = None
    total: Optional[str] = None
    rawText: str
    detectedCategory: str
    suggestedDescription: str
    processingTime: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    ocr_ready: bool
    ocr_loading: bool
    version: str
    environment: str
    uptime_seconds: int
    error: Optional[str] = None


# ======================== OCR FUNCTIONS ========================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Попередня обробка зображення"""
    try:
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=7)

        return denoised
    except Exception as e:
        logger.error(f"Помилка preprocessing: {e}")
        return image


def is_number(text: str) -> bool:
    """Перевірка чи є текст числом"""
    clean = text.replace(' ', '').replace(',', '.').replace('грн', '').replace('₴', '')
    clean = re.sub(r'[^\d.]', '', clean)
    if not clean:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return digit_ratio > 0.4


def clean_number(text: str) -> str:
    """Витягти чисте число з тексту"""
    clean = re.sub(r'[^\d.,\s-]', '', text)
    clean = clean.strip().replace(' ', '')

    if ',' in clean and clean.count(',') == 1:
        parts = clean.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            clean = clean.replace(',', '.')

    return clean


def detect_category(items: List[dict]) -> str:
    """Визначити категорію на основі товарів"""
    all_text = ' '.join([item['name'].lower() for item in items])

    if any(word in all_text for word in [
        'хліб', 'молоко', 'сир', "м'ясо", 'овоч', 'фрукт', 'їжа',
        'хлеб', 'молоко', 'сыр', 'мясо', 'овощ'
    ]):
        return "Їжа"
    elif any(word in all_text for word in [
        'бензин', 'паливо', 'проїзд', 'квиток',
        'бензин', 'топливо', 'проезд', 'билет'
    ]):
        return "Транспорт"
    elif any(word in all_text for word in [
        'аптека', 'ліки', 'медикамент', 'таблетк',
        'аптека', 'лекарств', 'медикамент', 'таблетк'
    ]):
        return "Здоров'я"
    elif any(word in all_text for word in [
        'футболка', 'штани', 'взуття', 'одяг',
        'футболка', 'штаны', 'обувь', 'одежда'
    ]):
        return "Одяг"
    elif any(word in all_text for word in [
        'комунальн', 'електр', 'вода', 'газ',
        'коммунальн', 'электр', 'вода', 'газ'
    ]):
        return "Комунальні"
    else:
        return "Інше"


def generate_description(items: List[dict]) -> str:
    """Створити опис витрати"""
    if len(items) == 0:
        return "Чек (товари не розпізнано)"
    elif len(items) == 1:
        return f"Чек: {items[0]['name']}"
    elif len(items) <= 3:
        return f"Чек: {', '.join([item['name'] for item in items])}"
    else:
        first_two = ', '.join([items[0]['name'], items[1]['name']])
        return f"Чек: {first_two} та ще {len(items) - 2}"


async def process_receipt(image: np.ndarray) -> dict:
    """Головна функція обробки чеку"""
    import time
    start_time = time.time()

    # Перевірка готовності OCR
    if reader is None:
        if reader_loading:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OCR_INITIALIZING",
                    "message": "OCR моделі завантажуються. Зачекайте 30-60 секунд і спробуйте ще раз.",
                    "retry_after": 30
                }
            )
        elif reader_load_error:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OCR_LOAD_FAILED",
                    "message": f"Не вдалося завантажити OCR: {reader_load_error}",
                }
            )
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OCR_NOT_READY",
                    "message": "OCR не готовий. Спробуйте через хвилину.",
                }
            )

    try:
        # Попередня обробка
        processed = preprocess_image(image)

        # OCR (виконуємо в executor щоб не блокувати event loop)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, reader.readtext, processed)

        # Форматування результатів
        ocr_data = []
        for bbox, text, conf in results:
            x1 = min(p[0] for p in bbox)
            y1 = min(p[1] for p in bbox)
            x2 = max(p[0] for p in bbox)
            y2 = max(p[1] for p in bbox)

            ocr_data.append({
                'text': text,
                'confidence': float(conf),
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            })

        ocr_data.sort(key=lambda r: r['y'])

        # Парсинг структури чеку
        receipt = {
            'items': [],
            'suma': None,
            'pdv': None,
            'doSplaty': None,
            'discount': None,
            'total': None,
            'rawText': ' '.join([r['text'] for r in ocr_data])
        }

        for i, item in enumerate(ocr_data):
            text = item['text']
            text_upper = text.upper()
            y = item['y']
            x = item['x']

            numbers_on_line = [
                r for r in ocr_data
                if abs(r['y'] - y) < 30 and is_number(r['text']) and r['x'] > x
            ]

            if any(word in text_upper for word in ['СУМА', 'SUMA', 'СЧНА', 'СYMA', 'CYMА']):
                if numbers_on_line:
                    receipt['suma'] = clean_number(numbers_on_line[0]['text'])

            elif 'ПДВ' in text_upper or 'PDV' in text_upper or 'VAT' in text_upper:
                if numbers_on_line:
                    receipt['pdv'] = clean_number(numbers_on_line[0]['text'])

            elif any(word in text_upper for word in ['СПЛАТИ', 'СПЛАТ', 'СПЛАТІ', 'ОПЛАТ', 'CNЛАТ']):
                if numbers_on_line:
                    receipt['doSplaty'] = clean_number(numbers_on_line[0]['text'])

            elif 'ЗНИЖК' in text_upper or 'DISCOUNT' in text_upper:
                if numbers_on_line:
                    receipt['discount'] = clean_number(numbers_on_line[0]['text'])

            elif 'TOTAL' in text_upper:
                if numbers_on_line:
                    receipt['total'] = clean_number(numbers_on_line[0]['text'])

            elif x < 600 and len(text) > 3:
                letter_ratio = sum(c.isalpha() or c in 'іїєґ' for c in text) / max(len(text), 1)

                if letter_ratio > 0.5:
                    price = None
                    for num_item in numbers_on_line:
                        if num_item['x'] > x + 200:
                            price = clean_number(num_item['text'])
                            break

                    if price:
                        receipt['items'].append({
                            'name': text,
                            'price': price,
                            'confidence': item['confidence']
                        })

        receipt['detectedCategory'] = detect_category(receipt['items'])
        receipt['suggestedDescription'] = generate_description(receipt['items'])
        receipt['processingTime'] = round(time.time() - start_time, 2)

        return receipt

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Помилка обробки чеку: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")


# ======================== API ENDPOINTS ========================

@app.get("/")
async def root():
    """Головна сторінка API"""
    uptime = (datetime.now() - server_start_time).seconds

    return {
        "message": "Receipt OCR API for Finance Game App",
        "version": "2.1.0",
        "status": "online",
        "ocr_ready": reader is not None,
        "ocr_loading": reader_loading,
        "uptime_seconds": uptime,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Перевірка здоров'я сервера"""
    uptime = (datetime.now() - server_start_time).seconds

    return HealthResponse(
        status="healthy" if reader is not None else ("initializing" if reader_loading else "error"),
        ocr_ready=reader is not None,
        ocr_loading=reader_loading,
        version="2.1.0",
        environment=os.getenv("RENDER", "local"),
        uptime_seconds=uptime,
        error=reader_load_error
    )


@app.post("/api/scan-receipt", response_model=ReceiptResponse)
async def scan_receipt(file: UploadFile = File(...)):
    """
    Розпізнати чек з фото

    - **file**: Фото чеку (JPG, PNG, max 10MB)

    Повертає структуровані дані чеку
    """
    max_size = 10 * 1024 * 1024  # 10MB

    try:
        contents = await file.read()

        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Файл завеликий. Максимум 10MB, отримано {len(contents) // 1024 // 1024}MB"
            )

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Не вдалося прочитати зображення")

        receipt = await process_receipt(image)

        return ReceiptResponse(
            success=True,
            items=[
                ReceiptItem(
                    name=item['name'],
                    price=item['price'],
                    confidence=item['confidence']
                ) for item in receipt['items']
            ],
            suma=receipt['suma'],
            pdv=receipt['pdv'],
            doSplaty=receipt['doSplaty'],
            discount=receipt['discount'],
            total=receipt['total'],
            rawText=receipt['rawText'],
            detectedCategory=receipt['detectedCategory'],
            suggestedDescription=receipt['suggestedDescription'],
            processingTime=receipt.get('processingTime')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Помилка API: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")


@app.get("/api/status")
async def api_status():
    """Детальний статус API"""
    uptime = (datetime.now() - server_start_time).seconds

    return {
        "server": {
            "status": "running",
            "uptime_seconds": uptime,
            "version": "2.1.0"
        },
        "ocr": {
            "ready": reader is not None,
            "loading": reader_loading,
            "error": reader_load_error
        },
        "endpoints": {
            "health": "/health",
            "scan": "/api/scan-receipt",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Запуск Receipt OCR API на порті {port}...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=120
    )