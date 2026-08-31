# ============================================================
# run_analysis.ps1  –  Week 1 EDA Runner
# Finds Python, installs dependencies, and runs the analysis.
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Try to find Python
$PythonCandidates = @(
    "python",
    "python3",
    "py",
    "C:\Program Files\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "C:\Users\hp\anaconda3\python.exe",
    "C:\ProgramData\Anaconda3\python.exe"
)

$PythonExe = $null
foreach ($candidate in $PythonCandidates) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python") {
            $PythonExe = $candidate
            Write-Host "Found Python: $candidate ($ver)" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $PythonExe) {
    Write-Host "ERROR: Python not found. Please install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Install requirements
Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
& $PythonExe -m pip install -r "$ScriptDir\requirements.txt" --quiet

# Run analysis
Write-Host "`nRunning EDA analysis..." -ForegroundColor Cyan
& $PythonExe "$ScriptDir\eda_analysis.py"

Write-Host "`nDone! Check the Week1_EDA folder for outputs." -ForegroundColor Green
