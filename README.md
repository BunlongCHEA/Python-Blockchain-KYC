```bash
python_kyc_service/
├── main.py               ← FastAPI app entry point + route registration, /health
├── config.py             ← Env / settings
├── database.py           ← PostgreSQL connection & queries
├── models.py             ← Pydantic request/response schemas
├── ocr/
│   ├── __init__.py
│   ├── reader.py         ← EasyOCR loader + image preprocessing
│   ├── extractor_id.py   ← Cambodian National ID field extraction
│   └── extractor_passport.py  ← Passport / MRZ field extraction
├── face/
│   ├── __init__.py
│   └── verify.py         ← DeepFace face comparison logic
├── routers/
│   ├── __init__.py
│   ├── scan.py           ← /api/kyc/scan routes
│   ├── face.py           ← /api/kyc/face routes
│   └── verify.py         ← /api/kyc/verify routes
├── utils/
│   ├── __init__.py
│   ├── image.py          ← base64 ↔ ndarray helpers
│   └── scoring.py        ← overall KYC score computation
├── Dockerfile
├── requirements.txt
└── .env.example
```

**Note:** __init__.py -- marks a regular folder as a Python package so that files inside it can be imported using dot notation like from ocr.reader import run_ocr.

❌ What Python sees without __init__.py

```bash
python_kyc_service/
├── ocr/
│   └── reader.py        ← Python does NOT know this exists as a module
```

If you try:

```bash
from ocr.reader import run_ocr   # ❌ ModuleNotFoundError
```

Python says "I don't know what ocr is" — it's just a plain folder.

✅ What Python sees WITH __init__.py

```bash
python_kyc_service/
├── ocr/
│   ├── __init__.py      ← "Hey Python, treat this folder as a package"
│   └── reader.py        ← now importable as ocr.reader
```

Now this works:

```bash
Python
from ocr.reader import run_ocr   # ✅ works perfectly
```