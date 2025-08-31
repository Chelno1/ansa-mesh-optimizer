@echo off
rem Simple Windows exe build script for ansa-mesh-optimizer
rem Fixed encoding issues - English only version

echo.
echo Building ansa-mesh-optimizer Windows exe
echo ========================================

rem Check if we're in the correct directory
if not exist "src\main.py" (
    echo ERROR: Please run this script from the project root directory
    echo The project root should contain src\main.py file
    pause
    exit /b 1
)

if not exist "pyproject.toml" (
    echo ERROR: pyproject.toml not found
    echo Make sure you're in the correct project directory
    pause
    exit /b 1
)

echo Current directory: %cd%
echo.

rem Check Python
echo Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to system PATH
    pause
    exit /b 1
)

echo Python check passed
echo.

rem Check pip
echo Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip not found, trying python -m pip
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: pip is not available
        pause
        exit /b 1
    )
    set PIP_CMD=python -m pip
) else (
    set PIP_CMD=pip
)

echo pip check passed
echo.

rem Install PyInstaller
echo Installing PyInstaller...
%PIP_CMD% install pyinstaller
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyInstaller
    echo Please check your network connection
    pause
    exit /b 1
)

echo.

rem Install project dependencies
echo Installing project dependencies...
%PIP_CMD% install -e .
if %errorlevel% neq 0 (
    echo Warning: Project dependency install failed, trying core packages...
    %PIP_CMD% install numpy matplotlib scikit-optimize scipy pandas seaborn psutil joblib tqdm colorama
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install core dependencies
        pause
        exit /b 1
    )
)

echo Dependencies installed successfully
echo.

rem Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build >nul 2>&1
if exist dist rmdir /s /q dist >nul 2>&1

rem Check spec file
if not exist "build_exe.spec" (
    echo ERROR: build_exe.spec not found
    echo Make sure the spec file is in the project root
    pause
    exit /b 1
)

echo.
echo Starting PyInstaller build...
echo This may take 5-10 minutes, please wait...
echo.

pyinstaller build_exe.spec
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
rem Check result
if exist "dist\ansa-mesh-optimizer.exe" (
    echo SUCCESS: Build completed!
    echo.
    echo File location: %cd%\dist\ansa-mesh-optimizer.exe
    
    for %%I in ("dist\ansa-mesh-optimizer.exe") do (
        echo File size: %%~zI bytes
    )
    
    echo.
    echo Testing exe file...
    "dist\ansa-mesh-optimizer.exe" --version
    if %errorlevel% equ 0 (
        echo exe file test passed!
    ) else (
        echo Warning: exe file may have issues
    )
    
) else (
    echo ERROR: Build failed - exe file not found
)

echo.
echo Build process completed
echo Press any key to exit...
pause >nul