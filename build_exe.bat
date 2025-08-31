@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ====================================================================
REM ansa-mesh-optimizer Windows exe 打包脚本
REM 版本: 0.2.0
REM 作者: Chel
REM 创建日期: 2025-08-30
REM 修复: 2025-08-31 - 解决窗口闪退问题
REM ====================================================================

echo.
echo 🚀 开始构建 ansa-mesh-optimizer Windows exe
echo =============================================
echo.

REM 显示当前目录
echo 📂 当前工作目录: %cd%
echo.

REM 检查当前目录是否为项目根目录
echo 🔍 检查项目文件结构...
if not exist "src\main.py" (
    echo.
    echo ❌ 错误：请在项目根目录运行此脚本
    echo    项目根目录应包含 src\main.py 文件
    echo.
    echo 💡 当前目录内容:
    dir /b
    echo.
    echo 请确保在包含以下文件的目录中运行:
    echo    - src\main.py
    echo    - pyproject.toml
    echo    - build_exe.spec
    echo.
    pause
    exit /b 1
)

if not exist "pyproject.toml" (
    echo.
    echo ❌ 错误：未找到 pyproject.toml 文件
    echo    请确保在正确的项目目录中运行
    echo.
    echo 💡 当前目录内容:
    dir /b
    echo.
    pause
    exit /b 1
)

echo ✅ 项目文件结构检查通过
echo.

REM 检查Python环境
echo 📋 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    echo    请安装Python 3.8+并添加到系统PATH
    pause
    exit /b 1
)

REM 显示Python版本
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python版本: %PYTHON_VERSION%

REM 检查pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip未安装或不可用
    pause
    exit /b 1
)

REM 升级pip
echo 📦 升级pip...
python -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo ⚠️  pip升级失败，继续使用当前版本
)

REM 安装PyInstaller
echo 📦 安装/升级PyInstaller...
pip install --upgrade pyinstaller --quiet
if %errorlevel% neq 0 (
    echo ❌ PyInstaller安装失败
    pause
    exit /b 1
)

REM 显示PyInstaller版本
for /f "tokens=2" %%i in ('pyinstaller --version 2^>^&1') do set PYINSTALLER_VERSION=%%i
echo ✅ PyInstaller版本: %PYINSTALLER_VERSION%

REM 安装项目依赖
echo 📚 安装项目依赖...
pip install -e . --quiet
if %errorlevel% neq 0 (
    echo ❌ 项目依赖安装失败
    echo    请检查pyproject.toml文件是否正确
    pause
    exit /b 1
)

echo ✅ 依赖安装完成

REM 清理之前的构建
echo 🧹 清理之前的构建文件...
if exist build (
    echo    删除 build 目录...
    rmdir /s /q build >nul 2>&1
)
if exist dist (
    echo    删除 dist 目录...
    rmdir /s /q dist >nul 2>&1
)

REM 清理旧的spec文件(保留我们的build_exe.spec)
for %%f in (*.spec) do (
    if not "%%f"=="build_exe.spec" (
        echo    删除旧spec文件: %%f
        del "%%f" >nul 2>&1
    )
)

echo ✅ 清理完成
echo.

REM 检查spec文件
echo 📋 检查配置文件...
if not exist "build_exe.spec" (
    echo.
    echo ❌ 错误：未找到 build_exe.spec 配置文件
    echo    请确保配置文件在项目根目录
    echo.
    echo 💡 当前目录的.spec文件:
    dir *.spec /b 2>nul
    echo.
    pause
    exit /b 1
)

echo ✅ 找到配置文件: build_exe.spec
echo.

REM 最后测试项目是否正常运行
echo 🧪 最后测试项目运行状态...
python src/main.py --version >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo ❌ 项目无法正常运行
    echo    请先确保以下命令能正常执行:
    echo    python src/main.py --version
    echo.
    pause
    exit /b 1
)
echo ✅ 项目运行测试通过
echo.

REM 显示构建信息
echo ============================================
echo 🔨 开始构建exe文件
echo ============================================
echo    配置文件: build_exe.spec
echo    目标平台: Windows
echo    打包模式: 单文件exe
echo    预计时间: 5-10分钟
echo ============================================
echo.
echo ⏳ 正在执行PyInstaller，请耐心等待...
echo    (这个过程可能看起来没有响应，这是正常的)
echo.

REM 使用spec文件构建
pyinstaller build_exe.spec
set BUILD_RESULT=!errorlevel!

echo.
echo ============================================
if !BUILD_RESULT! neq 0 (
    echo ❌ 构建失败！
    echo ============================================
    echo.
    echo 🔍 可能的解决方案：
    echo    1. 检查所有依赖是否正确安装
    echo    2. 运行测试: python src/main.py --help
    echo    3. 查看上方的详细错误信息
    echo    4. 检查防病毒软件是否阻止了构建
    echo    5. 尝试以管理员身份运行此脚本
    echo    6. 检查磁盘空间是否足够(至少2GB)
    echo.
    echo 如需调试，可以运行: build_exe_debug.bat
    echo.
    pause
    exit /b 1
) else (
    echo ✅ PyInstaller执行完成
    echo ============================================
)

REM 检查构建结果
if exist "dist\ansa-mesh-optimizer.exe" (
    echo ✅ 构建成功！
    echo.
    
    REM 显示文件信息
    for %%I in ("dist\ansa-mesh-optimizer.exe") do (
        echo 📁 exe文件位置: %cd%\dist\ansa-mesh-optimizer.exe
        echo 📊 文件大小: %%~zI 字节 (约 %%~zI/1048576 MB^)
    )
    
    REM 基本功能测试
    echo.
    echo 🧪 执行基本功能测试...
    
    REM 测试版本命令
    echo    测试 --version 命令...
    "dist\ansa-mesh-optimizer.exe" --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ 版本命令测试通过
    ) else (
        echo    ⚠️  版本命令测试失败
    )
    
    REM 测试帮助命令
    echo    测试 --help 命令...
    "dist\ansa-mesh-optimizer.exe" --help >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ 帮助命令测试通过
    ) else (
        echo    ⚠️  帮助命令测试失败
    )
    
    echo.
    echo ✨ 打包完成！
    echo.
    echo 📋 使用说明：
    echo    1. exe文件位于 dist\ 目录
    echo    2. 可以直接复制到其他Windows电脑运行
    echo    3. 无需安装Python环境
    echo    4. 首次启动可能需要3-5秒解压时间
    echo.
    echo 🧪 快速测试命令：
    echo    dist\ansa-mesh-optimizer.exe --version
    echo    dist\ansa-mesh-optimizer.exe --help
    echo    dist\ansa-mesh-optimizer.exe info --check-deps
    echo.
    
) else (
    echo ❌ 构建失败：未找到生成的exe文件
    echo    请检查上方的错误信息
)

echo 按任意键退出...
pause >nul