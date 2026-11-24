param(
    [switch]$FastDemo
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontend = Join-Path $projectRoot "app\frontend"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found at $python"
}

if ($FastDemo) {
    $env:GST_GENERATOR_MODEL = "extractive-evidence-v1"
}

Write-Host "Starting GST Lens backend on http://127.0.0.1:8000"
$backend = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Starting GST Lens frontend on http://127.0.0.1:5173"
try {
    Set-Location -LiteralPath $frontend
    npm run dev
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Stop-Process -Id $backend.Id
    }
    Set-Location -LiteralPath $projectRoot
}
