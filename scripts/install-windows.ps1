$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $pythonPath)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.13 with its py launcher first.' }
    }
    & $pythonPath -c "import sys; assert sys.version_info[:2] == (3, 13), 'This wheel requires Python 3.13'"
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv has the wrong Python version.' }
    & $pythonPath -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & $pythonPath -m pip install 'https://github.com/turboderp-org/exllamav3/releases/download/v1.5.3/exllamav3-1.5.3%2Bcu128.torch2.10.0-cp313-cp313-win_amd64.whl'
    if ($LASTEXITCODE -ne 0) { throw 'ExLlamaV3 installation failed.' }
    & $pythonPath -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw 'Adapter installation failed.' }
    & $pythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed. Check NVIDIA driver and PyTorch.' }
    Write-Host 'Environment ready. Download model weights separately; see README.md.' -ForegroundColor Green
} finally {
    Pop-Location
}
