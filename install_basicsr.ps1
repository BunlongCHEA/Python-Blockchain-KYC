# ============================================================
# install_basicsr.ps1
# Fix and install basicsr - PyPI version is broken due to
# KeyError '__version__' in setup.py get_version function.
#
# Run from your project root:
#   .\install_basicsr.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Fixing and Installing basicsr          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# -- Step 1: Clone BasicSR --
$cloneDir = "BasicSR"

if (Test-Path $cloneDir) {
    Write-Host "[1/5] Removing existing $cloneDir folder..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $cloneDir
}

Write-Host "[1/5] Cloning BasicSR from GitHub..." -ForegroundColor Green
git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git $cloneDir

$setupFile = Join-Path $cloneDir "setup.py"
if (-not (Test-Path $setupFile)) {
    Write-Host "ERROR: Clone failed - setup.py not found" -ForegroundColor Red
    exit 1
}

# -- Step 2: Write the correct version.py with both __version__ and __gitsha__ --
Write-Host "[2/5] Writing basicsr/version.py..." -ForegroundColor Green

$versionFile = Join-Path (Join-Path $cloneDir "basicsr") "version.py"
$versionContent = @"
__version__ = "1.4.2"
__gitsha__ = "unknown"
"@
Set-Content -Path $versionFile -Value $versionContent -Encoding UTF8

Write-Host "       Created: $versionFile" -ForegroundColor Gray

# -- Step 3: Replace setup.py entirely with a working version --
Write-Host "[3/5] Replacing setup.py with fixed version..." -ForegroundColor Green

$newSetup = @"
import os
from setuptools import find_packages, setup

def get_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-')]
    return []

setup(
    name='basicsr',
    version='1.4.2',
    description='BasicSR patched build',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=get_requirements(),
)
"@

Set-Content -Path $setupFile -Value $newSetup -Encoding UTF8

# Also remove pyproject.toml if it exists - it can override setup.py
$pyprojectFile = Join-Path $cloneDir "pyproject.toml"
if (Test-Path $pyprojectFile) {
    Write-Host "       Removing pyproject.toml to prevent build override..." -ForegroundColor Gray
    Remove-Item -Force $pyprojectFile
}

Write-Host "       Replaced: $setupFile" -ForegroundColor Gray

# -- Step 4: Install basicsr from the fixed local copy --
Write-Host "[4/5] Installing basicsr from fixed source..." -ForegroundColor Green

Push-Location $cloneDir
try {
    pip install --no-build-isolation .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed" -ForegroundColor Red
        exit 1
    }
}
finally {
    Pop-Location
}

# -- Step 5: Verify installation --
Write-Host "[5/5] Verifying installation..." -ForegroundColor Green

# Also fix the installed version.py in site-packages in case pip overwrote it
python -c "import basicsr, os; vp=os.path.join(os.path.dirname(basicsr.__file__),'version.py'); open(vp,'w').write('__version__=`"1.4.2`"`n__gitsha__=`"unknown`"`n'); print('basicsr', basicsr.__version__)"

if ($LASTEXITCODE -ne 0) {
    # If the above failed, manually fix the installed version.py and retry
    Write-Host "       Fixing installed version.py..." -ForegroundColor Yellow
    python -c "import basicsr, os; vp=os.path.join(os.path.dirname(basicsr.__file__),'version.py'); f=open(vp,'w'); f.write('__version__ = \`"1.4.2\`"\n__gitsha__ = \`"unknown\`"\n'); f.close()"
    python -c "import basicsr; print('basicsr', basicsr.__version__)"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: basicsr import still failed" -ForegroundColor Red
        exit 1
    }
}

# -- Cleanup --
Write-Host ""
Write-Host "Cleaning up cloned folder..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $cloneDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " basicsr installed successfully!        " -ForegroundColor Green
Write-Host " Now run:                               " -ForegroundColor Green
Write-Host "   pip install -r requirements.txt      " -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green