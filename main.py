"""
KYC AI Verification Service — entry point.
Registers all routers and exposes a /health endpoint.
"""
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import scan, face, verify

app = FastAPI(
    title="KYC AI Verification API",
    description=(
        "ID Card / Passport OCR (Khmer & English) + "
        "DeepFace face verification service for Go-Blockchain-KYC"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(scan.router)
app.include_router(face.router)
app.include_router(verify.router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "ok",
        "service":   "KYC AI Verification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )