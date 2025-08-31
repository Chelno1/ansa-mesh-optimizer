@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ====================================================================
REM ansa-mesh-optimizer exe文件测试脚本
REM 版本: 0.2.0
REM 作者: Chel
REM 创建日期: 2025-08-30
REM ====================================================================

echo.
echo 🧪 测试 ansa-mesh-optimizer.exe 功能
echo =====================================

REM 检查exe文件是否存在
if not exist "dist\ansa-mesh-optimizer.exe" (
    echo ❌ 错误：未找到 dist\ansa-mesh-optimizer.exe
    echo    请先运行 build_exe.bat 进行打包
    pause
    exit /b 1
)

set EXE_PATH=dist\ansa-mesh-optimizer.exe

REM 显示exe文件信息
echo 📁 exe文件路径: %cd%\%EXE_PATH%
for %%I in ("%EXE_PATH%") do (
    echo 📊 文件大小: %%~zI 字节 (约 !%%~zI:~0,-6! MB)
)
echo.

REM 测试计数器
set PASS_COUNT=0
set FAIL_COUNT=0
set TOTAL_TESTS=8

echo 🔍 开始功能测试...
echo.

REM 测试1：版本命令
echo [1/%TOTAL_TESTS%] 测试 --version 命令...
"%EXE_PATH%" --version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ 版本命令测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ 版本命令测试失败
    set /a FAIL_COUNT+=1
)

REM 测试2：帮助命令
echo [2/%TOTAL_TESTS%] 测试 --help 命令...
"%EXE_PATH%" --help >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ 帮助命令测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ 帮助命令测试失败
    set /a FAIL_COUNT+=1
)

REM 测试3：info命令
echo [3/%TOTAL_TESTS%] 测试 info 命令...
"%EXE_PATH%" info >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ info命令测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ info命令测试失败
    set /a FAIL_COUNT+=1
)

REM 测试4：info --check-deps命令
echo [4/%TOTAL_TESTS%] 测试 info --check-deps 命令...
"%EXE_PATH%" info --check-deps >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ 依赖检查测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ 依赖检查测试失败
    set /a FAIL_COUNT+=1
)

REM 测试5：config命令
echo [5/%TOTAL_TESTS%] 测试 config 命令...
"%EXE_PATH%" config >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ config命令测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ config命令测试失败
    set /a FAIL_COUNT+=1
)

REM 测试6：test命令
echo [6/%TOTAL_TESTS%] 测试 test 命令...
"%EXE_PATH%" test >nul 2>&1
if !errorlevel! leq 1 (
    echo ✅ test命令测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ test命令测试失败
    set /a FAIL_COUNT+=1
)

REM 测试7：optimize help
echo [7/%TOTAL_TESTS%] 测试 optimize --help 命令...
"%EXE_PATH%" optimize --help >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ optimize帮助测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ optimize帮助测试失败
    set /a FAIL_COUNT+=1
)

REM 测试8：compare help
echo [8/%TOTAL_TESTS%] 测试 compare --help 命令...
"%EXE_PATH%" compare --help >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ compare帮助测试通过
    set /a PASS_COUNT+=1
) else (
    echo ❌ compare帮助测试失败
    set /a FAIL_COUNT+=1
)

echo.
echo 📊 测试结果统计:
echo    ✅ 通过: %PASS_COUNT%/%TOTAL_TESTS%
echo    ❌ 失败: %FAIL_COUNT%/%TOTAL_TESTS%

if %FAIL_COUNT% equ 0 (
    echo.
    echo 🎉 所有测试通过！exe文件工作正常
    echo.
    echo 💡 可以进行以下高级测试:
    echo    "%EXE_PATH%" optimize --optimizer bayesian --n-calls 3 --evaluator mock
    echo    "%EXE_PATH%" compare --optimizers bayesian random --n-calls 3 --evaluator mock
) else (
    echo.
    echo ⚠️  部分测试失败，可能需要检查:
    echo    1. 缺失的隐式导入模块
    echo    2. 数据文件路径问题
    echo    3. 第三方库兼容性问题
)

echo.
echo 📋 详细信息显示:
echo.
echo "--- 版本信息 ---"
"%EXE_PATH%" --version 2>nul
echo.
echo "--- 系统信息 ---"
"%EXE_PATH%" info 2>nul
echo.

echo 测试完成，按任意键退出...
pause >nul