# Setup Kaggle credentials from credentials folder
# Run this once to configure Kaggle CLI

Write-Host "🔧 Setting up Kaggle credentials..." -ForegroundColor Cyan

$credentialsPath = "credentials\kaggle.json"

if (-not (Test-Path $credentialsPath)) {
    Write-Host "❌ Error: credentials\kaggle.json not found" -ForegroundColor Red
    Write-Host "   Please ensure your Kaggle API key is in credentials\kaggle.json" -ForegroundColor Yellow
    exit 1
}

# Read credentials
$kaggleConfig = Get-Content $credentialsPath | ConvertFrom-Json

# Create .kaggle directory in user home
$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
if (-not (Test-Path $kaggleDir)) {
    New-Item -ItemType Directory -Path $kaggleDir -Force | Out-Null
    Write-Host "✅ Created $kaggleDir" -ForegroundColor Green
}

# Copy kaggle.json to standard location
$destPath = Join-Path $kaggleDir "kaggle.json"
Copy-Item $credentialsPath $destPath -Force

Write-Host "✅ Copied kaggle.json to $destPath" -ForegroundColor Green
Write-Host "✅ Kaggle CLI is now configured!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "  python kaggle_train.py --push-dataset --run" -ForegroundColor White
