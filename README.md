# I. Install Python Libs
## 1. Run Install Python Libs

```bash
pip install -r requirements.txt
```

## 2. Install PyTorch to support GPU

```bash
pip uninstall torch torchvision torchaudio -y
```

```bash
# Install with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#OR to find Compatible with your GPU

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python -c "import torch; print('Version:', torch.__version__); print('CUDA built:', torch.version.cuda); print('Available:', torch.cuda.is_available())"

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA built with:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```


# II. Structure Project and Details

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

## 1. Structure & Process Google Cloud Vision OCR

```bash
Original Image
     │
     ├──→ Pass 1: run_ocr(full_bytes, hints=["km","en"])
     │         → full_texts (Khmer names, labels, context)
     │         → extract_cambodian_id_fields(full_texts)
     │
     ├──→ detect_mrz_zone(id_img)  ← OpenCV crop
     │         │
     │         └──→ Pass 2: run_ocr_mrz(mrz_bytes, hints=["en"])
     │                   → mrz_texts (clean ASCII MRZ lines)
     │                   → extract_cambodian_id_fields(mrz_texts)
     │
     └──→ MERGE: MRZ fields override full-image fields
              │
              └──→ _validate_extracted_fields()
                       │
                  ┌────┴────┐
              all OK     missing required
                 │            │
                 ▼            ▼
           continue       return OCR_INCOMPLETE
           pipeline        (frontend retries)
```

More Detail info from above Diagram:

```bash
┌─────────────────────────────────────────┐
│            Original Image               │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │   UPPER ZONE (Khmer + labels)   │ ──→  OCR Pass 1: Full image
│  │   Name, DOB, Nationality, etc.  │      hints=["km","en"]
│  └─────────────────────────────────┘    │      → Khmer raw_text + label fields
│  ┌─────────────────────────────────┐    │
│  │   LOWER ZONE (MRZ — ASCII)      │ ──→  OCR Pass 2: Cropped MRZ
│  │   IDKHM0110522898<<<...         │      hints=["en"]  (ASCII only)
│  └─────────────────────────────────┘    │      → Clean MRZ lines
└─────────────────────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  MERGE RESULTS  │
            │                 │
            │  MRZ wins for:  │
            │   id_number     │
            │   date_of_birth │
            │   expiry_date   │
            │   sex           │
            │   nationality   │
            │   first_name    │
            │   last_name     │
            │                 │
            │  Full wins for: │
            │   Khmer names   │
            │   raw_text      │
            │   (all context) │
            └─────────────────┘
```

## 2. Structure & Process Verify Face - DeepFace

```bash
┌─────────────────────┐         ┌─────────────────────┐
│    ID CARD PHOTO     │         │       SELFIE         │
│                      │         │                      │
│  ┌──────────┐        │         │        ┌──────────┐  │
│  │  Small   │        │         │        │  Large   │  │
│  │  ~80×100 │ ← low  │         │        │ ~400×500 │  │
│  │  pixels  │   res   │         │        │  pixels  │  │
│  └──────────┘        │         │        └──────────┘  │
│  - Printed/laminated │         │  - Natural lighting   │
│  - Flat lighting     │         │  - 3D perspective     │
│  - Possibly faded    │         │  - Camera distortion  │
│  - Scanned through   │         │  - Different angle    │
│    camera lens       │         │  - Years older/younger│
└─────────────────────┘         └─────────────────────┘
         │                                │
         └──── Distance = 0.55 ───────────┘
               (barely passing 0.68 threshold)
```

This is detail process for Verify Face

```bash
  ID Card Photo                          Selfie Photo
       │                                      │
       ▼                                      ▼
┌─────────────────┐                  ┌──────────────────┐
│ Attempt 1       │                  │ Normalize        │
│ GFPGAN          │──── if fail ───▶ │ Brightness       │
│ (AI face restore│                  │ (shared across   │
│  neural net)    │                  │  attempts 1-3)   │
└────────┬────────┘                  └────────┬─────────┘
         │                                    │
         ▼                                    │
  ┌──────────────┐                            │
  │ DeepFace     │◄───────────────────────────┘
  │ ArcFace      │
  │ distance=?   │
  └──────┬───────┘
         │
         ▼
┌─────────────────┐
│ Attempt 2       │
│ Real-ESRGAN     │──── if fail ───▶ skip
│ (AI upscale)    │
│ + OpenCV enhance│
│ + normalize     │
└────────┬────────┘
         │
         ▼
  ┌──────────────┐
  │ DeepFace     │
  │ distance=?   │
  └──────┬───────┘
         │
         ▼
┌─────────────────┐
│ Attempt 3       │
│ OpenCV only     │
│ upscale +       │
│ CLAHE + denoise │
│ + sharpen +     │
│ normalize       │
└────────┬────────┘
         │
         ▼
  ┌──────────────┐
  │ DeepFace     │
  │ distance=?   │
  └──────┬───────┘
         │
         ▼
┌─────────────────┐
│ Attempt 4       │
│ RAW             │
│ (no processing) │
│                 │
│ id_img as-is    │──────────────────┐
│ selfie as-is    │──────��───────┐   │
└────────┬────────┘              │   │
         │                       │   │
         ▼                       ▼   ▼
  ┌──────────────────────────────────────┐
  │ DeepFace.verify(                     │
  │   img1_path = id_img (original)      │
  │   img2_path = selfie (original)      │
  │   model     = ArcFace                │
  │   detector  = RetinaFace             │
  │   enforce_detection = False          │
  │   align     = True                   │
  │ )                                    │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │ Result:                              │
  │   distance  = ~0.72 (worst of all)   │
  │   threshold = 0.68                   │
  │   verified  = False (0.72 > 0.68)    │
  │   label     = "raw"                  │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │ Compare all 4 attempts               │
  │ Pick LOWEST distance                 │
  │                                      │
  │  gfpgan_restored:     0.35  ← BEST  │
  │  realesrgan_enhanced: 0.42           │
  │  opencv_enhanced:     0.61           │
  │  raw:                 0.72  ← WORST  │
  │                                      │
  │ Winner: gfpgan_restored              │
  └──────────────────────────────────────┘
```

Process of Verify Face:

| Area | Before | After |
| :--- | :--- | :--- |
| **ID photo size** | Sent as-is (~80×100px from card) | `_upscale_if_small()` → scales up to 250px minimum |
| **ID photo quality** | Raw scanned-through-camera image | `_enhance_id_photo()` → denoise + CLAHE contrast + sharpen |
| **Brightness mismatch** | Studio ID vs. natural selfie lighting | `_normalize_brightness()` → both images normalized to same mean luminance |
| **Alignment** | `align` not set (default varies) | `align=True` explicitly → 5-point landmark alignment guaranteed |
| **Retry strategy** | Single attempt, take or leave | 2 attempts — enhanced first, raw fallback. Best (lowest distance) wins |
| **Manual face crop** | N/A | Not added — RetinaFace already handles this better internally |
| **Response** | No info about preprocessing | New "preprocessing" field shows which attempt won ("enhanced" or "raw") |

# III. CI/CD Architecture

Diagram for main.yml

```bash
GitHub Push → main
      │
      ▼
.github/workflows/main.yml
  ├─ workflow_dispatch: USE_GPU=false → builds CPU → ghcr.io/.../kyc-python-api:latest
  └─ workflow_dispatch: USE_GPU=true  → builds GPU → ghcr.io/.../kyc-python-api:latest-gpu
      │
      ▼  (image pushed to GHCR)
ArgoCD watches GitHub repo
      │
      ▼
k8s/argocd/app/kyc-python-application.yaml
  └─ syncs k8s/app/ → namespace: kyc-python
        ├─ 00-namespace.yaml
        ├─ 01-secrets.yaml      (placeholder — real values via kubectl/Sealed Secrets)
        ├─ 02-configmap.yaml    (USE_GPU, POSTGRES_HOST, etc.)
        ├─ 03-deployment.yaml   (pulls ghcr.io image, injects Secret as env)
        └─ 04-service.yaml      (ClusterIP :5001)
      │
      ▼
 setup-cluster               (skipped on PR)
  ├─ Gateway API CRDs
  ├─ Traefik kubernetesGateway=true
  └─ cert-manager
      │
      ▼
   deploy
  ├─ namespace, ghcr-secret, secrets
  ├─ kubectl apply -f k8s/argocd/app/   ← 05-gateway.yaml now works
  └─ ArgoCD sync
```

### Required GitHub Secrets

Go to Settings → Secrets and variables → Actions and add:

| Secret Name | Value |
| :--- | :--- |
| `KUBE_CONFIG_BASE64` | `cat ~/.kube/config \| base64 -w0` |
| `POSTGRES_PASSWORD` | Input DB password |
| `GOOGLE_CREDENTIALS_B64` | `cat service-account.json \| base64 -w0` (optional) |