@echo off
echo ===============================================
echo ANSA Mesh Optimizer Windows Build Script
echo ===============================================

:: Set variables
set PROJECT_NAME=ansa-mesh-optimizer
set VERSION=0.2.0
set BUILD_DIR=dist_windows
set SPEC_FILE=ansa_mesh_optimizer.spec

:: Clean previous builds
echo Cleaning previous build results...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%" 2>nul
if exist "dist" rmdir /s /q "dist" 2>nul
if exist "build" rmdir /s /q "build" 2>nul
if exist "build_temp" rmdir /s /q "build_temp" 2>nul

:: Check Python environment
echo.
echo Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

:: Check pip
echo Checking pip...
pip --version
if errorlevel 1 (
    echo ERROR: pip not found
    pause
    exit /b 1
)

:: Check PyInstaller
echo.
echo Checking PyInstaller...
python -c "import PyInstaller; print('PyInstaller version:', PyInstaller.__version__)" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: PyInstaller installation failed
        pause
        exit /b 1
    )
)

:: Install project dependencies
echo.
echo Installing project dependencies...
pip install -e .
if errorlevel 1 (
    echo ERROR: Project dependencies installation failed
    pause
    exit /b 1
)

:: Install additional packages for better compatibility
echo.
echo Installing additional packages for visualization...
pip install Pillow
if errorlevel 1 (
    echo WARNING: Pillow installation failed, continuing without PIL support
) else (
    echo SUCCESS: Pillow installed successfully
)

:: Try to install matplotlib if not present
pip install matplotlib
if errorlevel 1 (
    echo WARNING: matplotlib installation failed, continuing without full visualization support
) else (
    echo SUCCESS: matplotlib installation verified
)

:: Check essential packages
echo.
echo Checking essential dependencies...
python -c "import numpy, scipy, matplotlib, sklearn, pandas, seaborn; print('All essential packages installed successfully')"
if errorlevel 1 (
    echo ERROR: Essential packages missing, please check requirements.txt
    pause
    exit /b 1
)

:: Run PyInstaller
echo.
echo ===============================================
echo Starting build process, this may take several minutes...
echo ===============================================
pyinstaller "%SPEC_FILE%" --distpath "%BUILD_DIR%" --workpath "build_temp" --clean
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

:: Check build results
echo.
echo Checking build results...
if exist "%BUILD_DIR%\%PROJECT_NAME%.exe" (
    echo.
    echo ===============================================
    echo BUILD SUCCESSFUL!
    echo Executable location: %BUILD_DIR%\%PROJECT_NAME%.exe
    echo.
    echo File size: 
    for %%F in ("%BUILD_DIR%\%PROJECT_NAME%.exe") do echo   %%~zF bytes
    echo ===============================================
    
    :: Run basic tests
    echo.
    echo Running basic tests...
    echo Test 1: Version information
    "%BUILD_DIR%\%PROJECT_NAME%.exe" --version
    if errorlevel 1 (
        echo WARNING: Version test failed
    ) else (
        echo SUCCESS: Version test passed
    )
    
    echo.
    echo Test 2: Help information
    "%BUILD_DIR%\%PROJECT_NAME%.exe" --help >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Help test failed
    ) else (
        echo SUCCESS: Help test passed
    )
    
    echo.
    echo Test 3: System info check
    "%BUILD_DIR%\%PROJECT_NAME%.exe" info --check-deps >nul 2>&1
    if errorlevel 1 (
        echo WARNING: System info check may have issues
    ) else (
        echo SUCCESS: System info check passed
    )
    
) else (
    echo ===============================================
    echo BUILD FAILED!
    echo Executable not found: %BUILD_DIR%\%PROJECT_NAME%.exe
    echo Please check error messages above
    echo ===============================================
    pause
    exit /b 1
)

:: Create release package
echo.
echo Creating release package...
set ZIP_NAME=%PROJECT_NAME%_v%VERSION%_windows.zip
if exist "%ZIP_NAME%" del "%ZIP_NAME%"

:: Use PowerShell to create zip
powershell -command "try { Compress-Archive -Path '%BUILD_DIR%\*' -DestinationPath '%ZIP_NAME%' -Force; Write-Host 'Release package created successfully' } catch { Write-Host 'Release package creation failed'; exit 1 }"
if errorlevel 1 (
    echo WARNING: Release package creation failed, but exe file is available
) else (
    echo SUCCESS: Release package created: %ZIP_NAME%
)

:: Display final results
echo.
echo ===============================================
echo BUILD COMPLETED!
echo ===============================================
echo Output files:
echo   Executable: %BUILD_DIR%\%PROJECT_NAME%.exe
if exist "%ZIP_NAME%" echo   Release package: %ZIP_NAME%
echo.
echo Usage:
echo   1. Run directly: %BUILD_DIR%\%PROJECT_NAME%.exe --help
echo   2. Or extract release package to target computer
echo.
echo Notes:
echo   - Ensure target Windows system has sufficient disk space
echo   - First run may require administrator privileges
echo   - Supports Windows 10 and above
echo ===============================================

pause