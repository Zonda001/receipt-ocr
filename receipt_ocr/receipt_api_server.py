"""
receipt_api_server.py - REST API for mobile app

Install:
    pip install fastapi uvicorn python-multipart easyocr opencv-python pillow

Run:
    python receipt_api_server.py

Test:
    curl -X POST -F "image=@receipt.jpg" http://localhost:8000/scan
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
from datetime import datetime
import hashlib

# Initialize API
app = FastAPI(
    title="Receipt OCR API",
    version="1.0",
    description="Mobile-friendly receipt scanning API"
)

# CORS for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR (initialized once)
OCR_READER = None

# Simple cache
CACHE = {}
MAX_CACHE = 50


def init_ocr():
    """Initialize OCR on startup"""
    global OCR_READER
    print("Initializing EasyOCR...")
    try:
        import easyocr
        OCR_READER = easyocr.Reader(['en', 'uk', 'ru'], gpu=True, verbose=False)
        print("OCR Ready!")
    except Exception as e:
        print(f"ERROR loading OCR: {e}")


@app.on_event("startup")
async def startup():
    """Run on startup"""
    init_ocr()


def enhance_image(image):
    """Enhance image quality"""
    # PIL enhancement
    pil_img = Image.fromarray(image)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.3)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)

    # Back to numpy
    img = np.array(pil_img)

    # Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 10)

    # Denoise
    clean = cv2.fastNlMeansDenoising(binary, h=10)

    return clean


def structure_receipt(ocr_data):
    """Simple receipt structuring"""
    receipt = {
        'items': [],
        'total': None,
        'cash': None,
        'change': None
    }

    sorted_data = sorted(ocr_data, key=lambda x: x['position']['y'])

    for item in sorted_data:
        text_upper = item['text'].upper()
        y = item['position']['y']

        # Find number on same line
        def find_num(y_pos):
            for other in sorted_data:
                if abs(other['position']['y'] - y_pos) < 20:
                    if any(c.isdigit() for c in other['text']):
                        return other['text']
            return None

        if 'TOTAL' in text_upper:
            num = find_num(y)
            if num:
                receipt['total'] = num
        elif 'CASH' in text_upper:
            num = find_num(y)
            if num:
                receipt['cash'] = num
        elif 'CHANGE' in text_upper:
            num = find_num(y)
            if num:
                receipt['change'] = num
        else:
            # Might be item
            if item['position']['x'] < 300 and any(c.isalpha() for c in item['text']):
                price = find_num(y)
                receipt['items'].append({
                    'name': item['text'],
                    'price': price
                })

    return receipt


@app.get("/")
def root():
    """API info"""
    return {
        "name": "Receipt OCR API",
        "version": "1.0",
        "status": "online" if OCR_READER else "ocr_not_loaded",
        "endpoints": {
            "/scan": "POST - Scan receipt image",
            "/scan/simple": "POST - Get simple text list",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "ok",
        "ocr_loaded": OCR_READER is not None,
        "cache_size": len(CACHE)
    }


@app.post("/scan")
async def scan_receipt(image: UploadFile = File(...)):
    """
    Scan receipt - full details

    Returns:
        {
            "success": true,
            "receipt": {
                "items": [...],
                "total": "59,500",
                "cash": "100,000",
                "change": "40,500"
            },
            "raw_ocr": [...]
        }
    """
    if not OCR_READER:
        raise HTTPException(status_code=503, detail="OCR not initialized")

    try:
        # Read image
        contents = await image.read()

        # Check cache
        img_hash = hashlib.md5(contents).hexdigest()
        if img_hash in CACHE:
            return CACHE[img_hash]

        # Decode
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Enhance
        enhanced = enhance_image(img)

        # OCR
        results = OCR_READER.readtext(enhanced)

        # Format
        ocr_data = []
        for bbox, text, conf in results:
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

        # Structure
        receipt = structure_receipt(ocr_data)

        # Response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "text_count": len(ocr_data),
            "receipt": receipt,
            "raw_ocr": ocr_data
        }

        # Cache
        if len(CACHE) < MAX_CACHE:
            CACHE[img_hash] = response

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan/simple")
async def scan_simple(image: UploadFile = File(...)):
    """
    Simple scan - just text list

    Returns:
        {
            "success": true,
            "texts": ["Choco Devil", "63,638", "Total:", "59,500", ...]
        }
    """
    if not OCR_READER:
        raise HTTPException(status_code=503, detail="OCR not initialized")

    try:
        contents = await image.read()

        # Decode
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Enhance
        enhanced = enhance_image(img)

        # OCR (simple mode)
        results = OCR_READER.readtext(enhanced, detail=0)

        return {
            "success": True,
            "texts": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("RECEIPT OCR API SERVER")
    print("=" * 60)
    print("\nStarting server...")
    print("\nEndpoints:")
    print("  http://localhost:8000/")
    print("  http://localhost:8000/scan")
    print("  http://localhost:8000/scan/simple")
    print("  http://localhost:8000/health")
    print("\nTest with curl:")
    print('  curl -X POST -F "image=@receipt.jpg" http://localhost:8000/scan/simple')
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)