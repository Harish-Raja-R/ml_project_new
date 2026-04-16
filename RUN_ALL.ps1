# RUN_ALL.ps1
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "🚀 NeuroCollab v3.0 - Starting All UIs" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "C:\Users\Harish\Downloads\ml_max_project_v4\ml_max_project"
Set-Location $projectRoot

# Check if models exist
if (-not (Test-Path "models/best_model.pkl")) {
    Write-Host "⚠️  Models not found. Training..." -ForegroundColor Yellow
    python build_project_max.py
}

Write-Host ""
Write-Host "📱 Starting 3 UIs in separate terminals..." -ForegroundColor Green
Write-Host ""

# Terminal 1: Flask
Write-Host "1️⃣  Flask (http://localhost:5000)" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$projectRoot\flask_app'; python app.py`""

# Small delay
Start-Sleep -Seconds 2

# Terminal 2: Streamlit
Write-Host "2️⃣  Streamlit (http://localhost:8501)" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$projectRoot'; streamlit run streamlit_app.py`""

# Small delay
Start-Sleep -Seconds 2

# Terminal 3: Gradio
Write-Host "3️⃣  Gradio (http://localhost:7860)" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$projectRoot'; python gradio_app.py`""

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "✅ ALL UIs LAUNCHED!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Open these in your browser:" -ForegroundColor Yellow
Write-Host "  🌐 Flask:     http://localhost:5000" -ForegroundColor Cyan
Write-Host "  📊 Streamlit: http://localhost:8501" -ForegroundColor Cyan
Write-Host "  🤗 Gradio:    http://localhost:7860" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C in any terminal to stop that service" -ForegroundColor Gray