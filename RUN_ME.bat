@echo off
title Week 1 EDA – Running Analysis...
echo ============================================================
echo   Week 1: Data Acquisition, Cleaning and EDA
echo   Double-clicked to start!
echo ============================================================
echo.

:: Try to find Python
set PYTHON=

:: Check py.exe launcher with specific version
where py >nul 2>&1
if %errorlevel% == 0 (
    py -V:3.12 --version >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON=py -V:3.12
        echo Found: Python 3.12 via py launcher
        goto :found
    )
    py -V:3.14 --version >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON=py -V:3.14
        echo Found: Python 3.14 via py launcher
        goto :found
    )
    py --version >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON=py
        echo Found: Python via py launcher
        goto :found
    )
)

:: Try python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON=python3
    echo Found: python3
    goto :found
)

:: Try python
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON=python
    echo Found: python
    goto :found
)

:: Try common install paths
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
    echo Found: Python 3.12 at AppData
    goto :found
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
    echo Found: Python 3.11 at AppData
    goto :found
)
if exist "C:\Program Files\Python312\python.exe" (
    set PYTHON="C:\Program Files\Python312\python.exe"
    echo Found: Python 3.12 at Program Files
    goto :found
)

echo.
echo ERROR: Python not found on this system.
echo Please install Python from https://python.org (version 3.10 or higher)
echo Then run this batch file again.
echo.
pause
exit /b 1

:found
echo.
echo Installing dependencies (first run may take a minute)...
%PYTHON% -m pip install pandas numpy matplotlib seaborn python-docx requests --quiet
echo.
echo Running EDA analysis...
%PYTHON% run_all.py
echo.
echo ============================================================
echo   Check the Week1_EDA folder for all output files:
echo     titanic_raw.csv
echo     titanic_cleaned.csv
echo     visualizations\fig1_missing_heatmap.png
echo     visualizations\fig2_survival_overview.png
echo     visualizations\fig3_distributions.png
echo     visualizations\fig4_correlation_heatmap.png
echo     visualizations\fig5_age_violin_box.png
echo     visualizations\fig6_fare_vs_age_scatter.png
echo     Week1_EDA_Report.docx
echo     Week1_EDA_Report.html  (open in browser or Word)
echo ============================================================
echo.
pause
