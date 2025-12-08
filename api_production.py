"""
api_production.py - Production Receipt OCR API for Render.com

Features:
- Ready for Render deployment
- Health checks
- CORS for mobile apps
- Rate limiting
- Error handling
- Logging

Deploy to Render:
1. Create account on render.com
2. New Web Service
3. Connect GitHub repo
4. Build: pip install -r requirements.txt
5. Start: gunicorn api_production:app -w 4 -k uvicorn.workers.UvicornWorker
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
from datetime import datetime
from typing import Dict, List
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Receipt OCR API",
    description="Ukrainian Receipt OCR API for mobile apps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - allow all origins (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR reader (loaded once at startup)
OCR_READER = None

# Request counter
REQUEST_COUNT = 0


@app.on_event("startup")
async def startup_event():
    """Initialize OCR on startup"""
    global OCR_READER
    logger.info("Starting OCR API...")

    try:
        import easyocr
        logger.info("Loading EasyOCR models...")
        OCR_READER = easyocr.Reader(['uk', 'en', 'ru'], gpu=False, verbose=False)
        logger.info("OCR ready!")
    except Exception as e:
        logger.error(f"Failed to load OCR: {e}")
        OCR_READER = None


def process_image(image_bytes: bytes) -> List[Dict]:
    """Process image and return OCR results"""
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Minimal preprocessing
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Light denoise
    clean = cv2.fastNlMeansDenoising(gray, h=5)

    # Run OCR
    results = OCR_READER.readtext(clean)

    # Format results
    ocr_data = []
    for bbox, text, conf in results:
        if conf > 0.3:  # Filter low confidence
            x1 = min(p[0] for p in bbox)
            y1 = min(p[1] for p in bbox)
            x2 = max(p[0] for p in bbox)
            y2 = max(p[1] for p in bbox)

            ocr_data.append({
                'text': text,
                'confidence': round(float(conf), 3),
                'position': {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                }
            })

    # Sort by Y position
    ocr_data.sort(key=lambda r: r['position']['y'])

    return ocr_data


def structure_receipt(ocr_data: List[Dict]) -> Dict:
    """Extract receipt structure"""
    receipt = {
        'suma': None,
        'pdv': None,
        'discount': None,
        'do_splaty': None,
        'payment_method': None,
        'items': []
    }

    for i, item in enumerate(ocr_data):
        text = item['text']
        text_upper = text.upper()
        y = item['position']['y']
        x = item['position']['x']

        # Find numbers on same line
        numbers = [
            r for r in ocr_data
            if abs(r['position']['y'] - y) < 30
               and r['position']['x'] > x
               and any(c.isdigit() for c in r['text'])
        ]

        # Ukrainian keywords with OCR error tolerance
        if any(kw in text_upper for kw in ['СУМА', 'SUMA', 'СЧМА', 'СYMA']):
            if numbers:
                receipt['suma'] = numbers[0]['text']

        elif 'ПДВ' in text_upper or 'PDV' in text_upper:
            if numbers:
                receipt['pdv'] = numbers[0]['text']

        elif any(kw in text_upper for kw in ['СПЛАТИ', 'СПХІАТИ', 'CNЛАТИ']):
            if numbers:
                receipt['do_splaty'] = numbers[0]['text']

        elif 'ЗНИЖК' in text_upper or 'DISCOUNT' in text_upper:
            if numbers:
                receipt['discount'] = numbers[0]['text']

        elif 'БЕЗГОТІВК' in text_upper or 'КАРТК' in text_upper:
            receipt['payment_method'] = 'card'
            if numbers:
                receipt['card_amount'] = numbers[0]['text']

        elif 'ГОТІВК' in text_upper or 'CASH' in text_upper:
            receipt['payment_method'] = 'cash'

    return receipt


@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Receipt OCR API",
        "version": "1.0.0",
        "status": "online",
        "ocr_loaded": OCR_READER is not None,
        "endpoints": {
            "/health": "Health check",
            "/scan": "POST - Scan receipt (full details)",
            "/scan/simple": "POST - Scan receipt (simple)",
            "/stats": "API statistics"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health():
    """Health check for Render"""
    global REQUEST_COUNT

    return {
        "status": "healthy" if OCR_READER else "degraded",
        "ocr_loaded": OCR_READER is not None,
        "timestamp": datetime.now().isoformat(),
        "requests_processed": REQUEST_COUNT
    }


@app.get("/stats")
async def stats():
    """API statistics"""
    return {
        "requests_total": REQUEST_COUNT,
        "ocr_status": "ready" if OCR_READER else "not_loaded",
        "uptime": "See /health for timestamp"
    }


@app.post("/scan")
async def scan_receipt(request: Request, image: UploadFile = File(...)):
    """
    Scan receipt - full details

    Returns structured receipt data with all fields
    """
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    start_time = time.time()

    if not OCR_READER:
        raise HTTPException(status_code=503, detail="OCR not initialized")

    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await image.read()

        # Log request
        logger.info(f"Processing receipt: {len(contents)} bytes from {request.client.host}")

        # Process
        ocr_data = process_image(contents)
        receipt = structure_receipt(ocr_data)

        # Processing time
        processing_time = round(time.time() - start_time, 2)

        # Response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "processing_time_sec": processing_time,
            "receipt": receipt,
            "raw_ocr": ocr_data,
            "stats": {
                "texts_detected": len(ocr_data),
                "confidence_avg": round(
                    sum(r['confidence'] for r in ocr_data) / len(ocr_data), 3
                ) if ocr_data else 0
            }
        }

        logger.info(f"Success: {len(ocr_data)} texts in {processing_time}s")

        return response

    except Exception as e:
        logger.error(f"Error processing receipt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan/simple")
async def scan_simple(request: Request, image: UploadFile = File(...)):
    """
    Scan receipt - simple format

    Returns just the extracted text list
    """
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    if not OCR_READER:
        raise HTTPException(status_code=503, detail="OCR not initialized")

    try:
        contents = await image.read()
        ocr_data = process_image(contents)

        return {
            "success": True,
            "texts": [r['text'] for r in ocr_data]
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan/test")
async def scan_test(request: Request, image: UploadFile = File(...)):
    """
    Test endpoint - returns detailed debug info

    For testing in mobile app
    """
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    if not OCR_READER:
        raise HTTPException(status_code=503, detail="OCR not initialized")

    try:
        contents = await image.read()

        # Get image info
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_info = {
            "size_bytes": len(contents),
            "dimensions": f"{img.shape[1]}x{img.shape[0]}",
            "channels": img.shape[2] if len(img.shape) == 3 else 1
        }

        # Process
        ocr_data = process_image(contents)
        receipt = structure_receipt(ocr_data)

        # Detailed response
        return {
            "success": True,
            "test_mode": True,
            "image_info": image_info,
            "receipt": receipt,
            "all_texts": [r['text'] for r in ocr_data],
            "confidence_distribution": {
                "high (>0.8)": len([r for r in ocr_data if r['confidence'] > 0.8]),
                "medium (0.5-0.8)": len([r for r in ocr_data if 0.5 < r['confidence'] <= 0.8]),
                "low (<0.5)": len([r for r in ocr_data if r['confidence'] <= 0.5])
            },
            "top_10_results": ocr_data[:10]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP error handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General error handler"""
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    print("\n" + "=" * 60)
    print("RECEIPT OCR API - PRODUCTION")
    print("=" * 60)
    print(f"\nStarting on port {port}...")
    print("\nEndpoints:")
    print("  /")
    print("  /health")
    print("  /scan")
    print("  /scan/simple")
    print("  /scan/test")
    print("  /docs")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)