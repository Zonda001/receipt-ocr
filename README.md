# 🧾 Receipt OCR API - Українські Чеки

**Розпізнавання українських чеків для мобільного додатку**

![Status](https://img.shields.io/badge/status-production-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)

---

## 📋 Зміст

- [Про проєкт](#про-проєкт)
- [Можливості](#можливості)
- [Технології](#технології)
- [Швидкий старт](#швидкий-старт)
- [Deployment на Render](#deployment-на-render)
- [API Endpoints](#api-endpoints)
- [Kotlin Integration](#kotlin-integration)
- [Що ми зробили](#що-ми-зробили)
- [Над чим працюємо](#над-чим-працюємо)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Про проєкт

**Receipt OCR API** - це REST API для розпізнавання українських касових чеків з фотографій. 

Створено спеціально для інтеграції з мобільним додатком (Kotlin) для автоматичного обліку фінансів.

### Основні переваги

- ✅ **Підтримка української мови** (UK, EN, RU)
- ✅ **Працює з поганою якістю фото**
- ✅ **Швидко** (2-3 секунди на чек)
- ✅ **Production-ready** (готовий до deployment)
- ✅ **Free tier** (працює на Render.com безкоштовно)

---

## 🚀 Можливості

### Що розпізнає API

| Поле | Приклад | Підтримка |
|------|---------|-----------|
| **Сума чеку** | 64.98 ГРН | ✅ |
| **ПДВ** | 10.83 (20%) | ✅ |
| **До сплати** | 64.98 ГРН | ✅ |
| **Спосіб оплати** | Картка/Готівка | ✅ |
| **Знижка** | -42.50 | ✅ |
| **Товари** | Назва + ціна | ⚠️ (залежить від якості фото) |
| **Магазин** | Сільпо-Фуд | ✅ |
| **Дата/час** | 29.99.2024 | ✅ |

### Формати відповіді

**1. Повний** (`/scan`) - всі дані + raw OCR
**2. Простий** (`/scan/simple`) - тільки текст
**3. Тестовий** (`/scan/test`) - детальна діагностика

---

## 🛠 Технології

### Backend
- **FastAPI** - сучасний Python web framework
- **EasyOCR** - розпізнавання тексту
- **OpenCV** - обробка зображень
- **Gunicorn** - production WSGI server

### Deployment
- **Render.com** - безкоштовний хостинг
- **Docker** - containerization (опціонально)

### Mobile (Kotlin)
- **Retrofit** - HTTP client
- **Coroutines** - async operations
- **Coil** - image loading

---

## ⚡ Швидкий старт

### Локальна розробка

```bash
# 1. Clone repo
git clone https://github.com/your-repo/receipt-ocr
cd receipt-ocr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
python api_production.py

# 4. Test
curl -X POST -F "image=@receipt.jpg" http://localhost:8000/scan/simple
```

### Docker

```bash
# Build
docker build -t receipt-ocr .

# Run
docker run -p 8000:8000 receipt-ocr

# Test
curl http://localhost:8000/health
```

---

## 🌐 Deployment на Render

### Автоматичний deployment

1. **Створи акаунт на Render.com**
   - Відкрий https://render.com
   - Sign up (безкоштовно)

2. **New Web Service**
   - Dashboard → New → Web Service
   - Connect GitHub repo

3. **Налаштування:**
   ```
   Name: receipt-ocr-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn api_production:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
   ```

4. **Deploy!**
   - Натисни "Create Web Service"
   - Чекай 5-10 хвилин
   - Отримай URL: `https://receipt-ocr-api.onrender.com`

### Перевірка deployment

```bash
# Health check
curl https://receipt-ocr-api.onrender.com/health

# Має повернути:
{
  "status": "healthy",
  "ocr_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 📡 API Endpoints

### Base URL
```
Local: http://localhost:8000
Render: https://receipt-ocr-api.onrender.com
```

### Endpoints

#### `GET /`
Інформація про API
```bash
curl https://receipt-ocr-api.onrender.com/
```

#### `GET /health`
Health check (для моніторингу)
```bash
curl https://receipt-ocr-api.onrender.com/health
```

#### `POST /scan`
Повне сканування чеку
```bash
curl -X POST \
  -F "image=@receipt.jpg" \
  https://receipt-ocr-api.onrender.com/scan
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-15T10:30:00",
  "processing_time_sec": 2.5,
  "receipt": {
    "suma": "64.98",
    "pdv": "10.83",
    "do_splaty": "64.98",
    "payment_method": "card"
  },
  "raw_ocr": [...],
  "stats": {
    "texts_detected": 45,
    "confidence_avg": 0.85
  }
}
```

#### `POST /scan/simple`
Простий формат (тільки текст)
```bash
curl -X POST \
  -F "image=@receipt.jpg" \
  https://receipt-ocr-api.onrender.com/scan/simple
```

**Response:**
```json
{
  "success": true,
  "texts": ["СУМА", "64.98", "ПДВ", "10.83", ...]
}
```

#### `POST /scan/test`
Тестовий endpoint (детальна діагностика)
```bash
curl -X POST \
  -F "image=@receipt.jpg" \
  https://receipt-ocr-api.onrender.com/scan/test
```

**Response:**
```json
{
  "success": true,
  "test_mode": true,
  "image_info": {
    "size_bytes": 245632,
    "dimensions": "1920x2560",
    "channels": 3
  },
  "receipt": {...},
  "confidence_distribution": {
    "high (>0.8)": 32,
    "medium (0.5-0.8)": 18,
    "low (<0.5)": 5
  }
}
```

---

## 📱 Kotlin Integration

### 1. Додай dependencies

```gradle
// build.gradle.kts
dependencies {
    // Retrofit
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    
    // OkHttp
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Coil (image loading)
    implementation("io.coil-kt:coil:2.5.0")
}
```

### 2. API Service

```kotlin
// ReceiptApi.kt
interface ReceiptApi {
    @Multipart
    @POST("/scan")
    suspend fun scanReceipt(
        @Part image: MultipartBody.Part
    ): ReceiptResponse
    
    @Multipart
    @POST("/scan/simple")
    suspend fun scanSimple(
        @Part image: MultipartBody.Part
    ): SimpleResponse
    
    @Multipart
    @POST("/scan/test")
    suspend fun scanTest(
        @Part image: MultipartBody.Part
    ): TestResponse
}

// Data classes
data class ReceiptResponse(
    val success: Boolean,
    val timestamp: String,
    val processing_time_sec: Double,
    val receipt: Receipt,
    val raw_ocr: List<OcrResult>,
    val stats: Stats
)

data class Receipt(
    val suma: String?,
    val pdv: String?,
    val do_splaty: String?,
    val payment_method: String?,
    val discount: String?
)

data class OcrResult(
    val text: String,
    val confidence: Double,
    val position: Position
)

data class Position(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int
)

data class Stats(
    val texts_detected: Int,
    val confidence_avg: Double
)
```

### 3. Retrofit Client

```kotlin
// RetrofitClient.kt
object RetrofitClient {
    private const val BASE_URL = "https://receipt-ocr-api.onrender.com/"
    
    private val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    
    private val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    
    val api: ReceiptApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ReceiptApi::class.java)
    }
}
```

### 4. ViewModel

```kotlin
// ReceiptViewModel.kt
class ReceiptViewModel : ViewModel() {
    private val _scanResult = MutableLiveData<Result<ReceiptResponse>>()
    val scanResult: LiveData<Result<ReceiptResponse>> = _scanResult
    
    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading
    
    fun scanReceipt(imageUri: Uri, context: Context) {
        viewModelScope.launch {
            _isLoading.value = true
            
            try {
                // Convert URI to file
                val file = File(context.cacheDir, "receipt_${System.currentTimeMillis()}.jpg")
                context.contentResolver.openInputStream(imageUri)?.use { input ->
                    file.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                
                // Create multipart body
                val requestBody = file.asRequestBody("image/jpeg".toMediaType())
                val imagePart = MultipartBody.Part.createFormData(
                    "image",
                    file.name,
                    requestBody
                )
                
                // Call API
                val response = RetrofitClient.api.scanReceipt(imagePart)
                _scanResult.value = Result.success(response)
                
                // Cleanup
                file.delete()
                
            } catch (e: Exception) {
                _scanResult.value = Result.failure(e)
            } finally {
                _isLoading.value = false
            }
        }
    }
}
```

### 5. UI Fragment/Activity

```kotlin
// ScanReceiptFragment.kt
class ScanReceiptFragment : Fragment() {
    private val viewModel: ReceiptViewModel by viewModels()
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { scanReceipt(it) }
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Observe results
        viewModel.scanResult.observe(viewLifecycleOwner) { result ->
            result.onSuccess { response ->
                displayReceipt(response.receipt)
            }.onFailure { error ->
                showError(error.message)
            }
        }
        
        // Scan button
        binding.btnScan.setOnClickListener {
            pickImage.launch("image/*")
        }
    }
    
    private fun scanReceipt(uri: Uri) {
        viewModel.scanReceipt(uri, requireContext())
    }
    
    private fun displayReceipt(receipt: Receipt) {
        binding.apply {
            tvSuma.text = receipt.suma ?: "N/A"
            tvPdv.text = receipt.pdv ?: "N/A"
            tvTotal.text = receipt.do_splaty ?: "N/A"
            tvPayment.text = when(receipt.payment_method) {
                "card" -> "Картка"
                "cash" -> "Готівка"
                else -> "Не визначено"
            }
        }
    }
}
```

### 6. Beta Test Window

```kotlin
// BetaTestActivity.kt
class BetaTestActivity : AppCompatActivity() {
    private lateinit var binding: ActivityBetaTestBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBetaTestBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Test endpoints
        binding.btnTestHealth.setOnClickListener { testHealth() }
        binding.btnTestScan.setOnClickListener { testScan() }
        binding.btnTestSimple.setOnClickListener { testSimple() }
    }
    
    private fun testHealth() {
        lifecycleScope.launch {
            try {
                val response = RetrofitClient.api.health()
                binding.tvResult.text = """
                    Status: ${response.status}
                    OCR Loaded: ${response.ocr_loaded}
                    Timestamp: ${response.timestamp}
                """.trimIndent()
            } catch (e: Exception) {
                binding.tvResult.text = "Error: ${e.message}"
            }
        }
    }
}
```

---

## ✅ Що ми зробили

### Phase 1: OCR Engine (✅ Completed)
- [x] EasyOCR integration
- [x] Українська мова підтримка
- [x] Image preprocessing
- [x] Smart text detection
- [x] Field extraction (СУМА, ПДВ, тощо)

### Phase 2: API Development (✅ Completed)
- [x] FastAPI REST API
- [x] Multiple endpoints (/scan, /scan/simple, /scan/test)
- [x] Error handling
- [x] CORS для мобільних додатків
- [x] Health checks
- [x] Logging

### Phase 3: Production Ready (✅ Completed)
- [x] Render.com deployment config
- [x] Dockerfile
- [x] Requirements.txt
- [x] Environment variables
- [x] Production logging

### Phase 4: Documentation (✅ Completed)
- [x] API documentation (/docs)
- [x] README.md
- [x] Kotlin integration examples
- [x] Deployment guide

---

## 🚧 Над чим працюємо

### Phase 5: Kotlin App Integration (🔄 In Progress)
- [ ] Complete Kotlin SDK
- [ ] Beta test window in app
- [ ] Beautiful UI for results
- [ ] Offline mode (cached results)
- [ ] Receipt history

### Phase 6: Accuracy Improvements (📋 Planned)
- [ ] Fine-tune OCR for receipts
- [ ] Better item detection
- [ ] Multi-receipt batch processing
- [ ] Receipt validation rules

### Phase 7: Advanced Features (💡 Ideas)
- [ ] Receipt categorization (продукти, ресторан, тощо)
- [ ] Budget tracking
- [ ] Analytics dashboard
- [ ] Export to Excel/PDF
- [ ] Cloud backup

---

## 🐛 Troubleshooting

### API не відповідає

```bash
# Check health
curl https://receipt-ocr-api.onrender.com/health

# Check logs on Render
# Dashboard → Your Service → Logs
```

### Низька точність розпізнавання

**Причини:**
- Погана якість фото
- Низьке освітлення
- Розмитість
- Нахил чеку

**Рішення:**
- Фотографуй при хорошому освітленні
- Тримай камеру прямо
- Без розмиття
- Роздільність мінімум 1280x720

### Timeout errors

Render free tier засинає після 15 хвилин неактивності.

**Рішення:**
- Перший запит може бути повільним (30-60 сек)
- Додай retry логіку в Kotlin
- Upgrade to paid plan для production

### CORS errors

Переконайся що в API дозволені твої origins:

```python
# api_production.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com"],  # Твій домен
    ...
)
```

---

## 📞 Контакти

- GitHub: [your-repo]
- Issues: [github.com/your-repo/issues]
- Email: your@email.com

---

## 📄 Ліцензія

MIT License - використовуй як хочеш!

---

**Made with ❤️ for Ukrainian receipt processing**