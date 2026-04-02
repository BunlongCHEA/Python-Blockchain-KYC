# ============================================================
# download_models.ps1
# Download GFPGAN and RealESRGAN pre-trained model weights.
#
# Run from your Python-Blockchain-KYC project root:
#   .\download_models.ps1
# ============================================================

$ErrorActionPreference = "Stop"

# Create models directory
$modelsDir = "models_pretrained"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Downloading face restoration models    " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# -- GFPGAN v1.4 (~350MB) --
$gfpganUrl = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
$gfpganPath = Join-Path $modelsDir "GFPGANv1.4.pth"

if (Test-Path $gfpganPath) {
    Write-Host "[1/2] GFPGANv1.4.pth already exists - skipping" -ForegroundColor Yellow
} else {
    Write-Host "[1/2] Downloading GFPGANv1.4.pth (~350MB)..." -ForegroundColor Green
    Invoke-WebRequest -Uri $gfpganUrl -OutFile $gfpganPath
    Write-Host "       Saved: $gfpganPath" -ForegroundColor Gray
}

# -- RealESRGAN x2plus (~64MB) --
$esrganUrl = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
$esrganPath = Join-Path $modelsDir "RealESRGAN_x2plus.pth"

if (Test-Path $esrganPath) {
    Write-Host "[2/2] RealESRGAN_x2plus.pth already exists - skipping" -ForegroundColor Yellow
} else {
    Write-Host "[2/2] Downloading RealESRGAN_x2plus.pth (~64MB)..." -ForegroundColor Green
    Invoke-WebRequest -Uri $esrganUrl -OutFile $esrganPath
    Write-Host "       Saved: $esrganPath" -ForegroundColor Gray
}

# -- Verify --
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Models downloaded successfully!        " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Add to your .env:" -ForegroundColor White
Write-Host "  GFPGAN_MODEL_PATH=models_pretrained/GFPGANv1.4.pth" -ForegroundColor Cyan
Write-Host "  REALESRGAN_MODEL_PATH=models_pretrained/RealESRGAN_x2plus.pth" -ForegroundColor Cyan