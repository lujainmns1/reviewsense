# ReviewSense Flask Backend

خادم Flask بسيط لتحليل المراجعات باستخدام Google Gemini AI.

## 🚀 التشغيل السريع

1. **تثبيت التبعيات:**
   ```bash
   pip install -r requirements.txt
   ```

2. **إعداد مفتاح API:**
   - افتح ملف `.env`
   - استبدل `your_gemini_api_key_here` بمفتاح Gemini API الحقيقي

3. **تشغيل الخادم:**
   ```bash
   python app.py
   ```

4. **التحقق من العمل:**
   - افتح: http://localhost:5000/health

## 📡 API Endpoints

- `GET /` - معلومات API
- `GET /health` - فحص حالة الخادم
- `POST /analyze` - تحليل المراجعات

## 📝 مثال على الاستخدام

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["منتج رائع!", "خدمة سيئة"]}'
```

## installing docker and creating container for postrgresql 
```bash
docker run -d --name reviewsense-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=reviewsense -p 5433:5432 postgres
```