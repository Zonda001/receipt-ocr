---
title: Receipt OCR API 🇺🇦
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Receipt OCR API 🧾🇺🇦

**API для розпізнавання українських чеків** з використанням EasyOCR та FastAPI.

## 🎯 Можливості

- ✅ **Українська мова** - розпізнає кирилицю без проблем
- ✅ **Автоматичне виявлення товарів** - назва + ціна
- ✅ **Фінансові підсумки** - сума, ПДВ, знижки, до сплати
- ✅ **Швидка обробка** - оптимізовано для продакшн
- ✅ **REST API** - легка інтеграція з мобільними додатками

## 📡 API Endpoints

### 1. Розпізнати чек (multipart)
```bash
POST /api/ocr
Content-Type: multipart/form-data

curl -X POST https://YOUR_SPACE_URL/api/ocr \
  -F "file=@receipt.jpg"
```

### 2. Розпізнати чек (base64)
```bash
POST /api/ocr/base64
Content-Type: application/json

{
  "image": "base64_encoded_image...",
  "include_raw": false
}
```

### 3. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "ocr_ready": true
}
```

## 📱 Приклади використання

### Flutter
```dart
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map> scanReceipt(File image) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('https://YOUR_SPACE_URL/api/ocr')
  );
  request.files.add(await http.MultipartFile.fromPath('file', image.path));
  
  var response = await request.send();
  var data = await response.stream.bytesToString();
  return json.decode(data);
}
```

### JavaScript/React Native
```javascript
const scanReceipt = async (imageUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'receipt.jpg'
  });
  
  const response = await fetch('https://YOUR_SPACE_URL/api/ocr', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

### Python
```python
import requests

with open('receipt.jpg', 'rb') as f:
    response = requests.post(
        'https://YOUR_SPACE_URL/api/ocr',
        files={'file': f}
    )
    
data = response.json()
print(f"Товарів: {len(data['receipt']['items'])}")
print(f"Сума: {data['receipt']['suma']}")
```

## 📊 Приклад відповіді

```json
{
  "success": true,
  "receipt": {
    "items": [
      {
        "name": "Хліб білий",
        "price": "25.50",
        "confidence": 0.92
      },
      {
        "name": "Молоко 2.5%",
        "price": "38.90",
        "confidence": 0.95
      }
    ],
    "suma": "64.40",
    "pdv": "10.73",
    "do_splaty": "64.40",
    "bezgotivkova": "64.40"
  },
  "meta": {
    "filename": "receipt.jpg",
    "text_regions_found": 45,
    "items_found": 2
  }
}
```

## 🛠 Технології

- **EasyOCR** - розпізнавання тексту
- **FastAPI** - REST API фреймворк
- **OpenCV** - обробка зображень
- **Python 3.10** - runtime

## 📝 Ліцензія

MIT License - використовуй вільно!

## 🤝 Контрібуція

Issues та Pull Requests вітаються!

---

**Створено для українських розробників 🇺🇦**