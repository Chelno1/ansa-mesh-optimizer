#!/bin/bash

# ==========================================
# ANSA Mesh Optimizer Linux打包脚本（兼容版）
# 
# 功能：
# - 创建独立的虚拟环境
# - 安装兼容的依赖版本（NumPy 2.0.2）
# - 使用PyInstaller打包为单文件可执行程序
# - 生成SHA256校验和
# 
# 使用方法：
# chmod +x build_linux_compatible.sh
# ./build_linux_compatible.sh
# ==========================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    print_info "检查Python版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "未找到Python解释器"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    print_success "找到Python版本: $PYTHON_VERSION"
    
    # 检查版本是否满足要求（至少3.8）
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $MAJOR_VERSION -lt 3 ]] || [[ $MAJOR_VERSION -eq 3 && $MINOR_VERSION -lt 8 ]]; then
        print_error "Python版本必须>=3.8，当前版本: $PYTHON_VERSION"
        exit 1
    fi
}

# 创建并激活虚拟环境
setup_venv() {
    print_info "创建虚拟环境..."
    
    # 如果虚拟环境已存在，先删除
    if [ -d "venv_build" ]; then
        print_warning "删除旧的虚拟环境..."
        rm -rf venv_build
    fi
    
    # 创建新的虚拟环境
    $PYTHON_CMD -m venv venv_build
    print_success "虚拟环境创建成功"
    
    # 激活虚拟环境
    source venv_build/bin/activate
    print_success "虚拟环境已激活"
    
    # 升级pip
    print_info "升级pip..."
    pip install --upgrade pip setuptools wheel
}

# 安装依赖
install_dependencies() {
    print_info "安装项目依赖（使用兼容版本）..."
    
    # 使用兼容的requirements文件
    if [ -f "requirements_build.txt" ]; then
        pip install -r requirements_build.txt
        print_success "依赖安装完成"
    else
        print_error "未找到requirements_build.txt文件"
        exit 1
    fi
    
    # 安装PyInstaller
    print_info "安装PyInstaller..."
    pip install pyinstaller==6.8.0
    print_success "PyInstaller安装完成"
}

# 清理旧的构建文件
clean_build() {
    print_info "清理旧的构建文件..."
    
    # 删除之前的构建目录
    rm -rf build dist *.egg-info __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "清理完成"
}

# 运行打包
run_pyinstaller() {
    print_info "开始打包..."
    
    # 设置环境变量以避免OpenBLAS问题
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    
    # 运行PyInstaller
    if [ -f "ansa_optimizer_linux_final.spec" ]; then
        pyinstaller ansa_optimizer_linux_final.spec --clean --noconfirm
        print_success "打包完成"
    else
        print_error "未找到ansa_optimizer_linux_final.spec文件"
        exit 1
    fi
}

# 测试可执行文件
test_executable() {
    print_info "测试生成的可执行文件..."
    
    if [ -f "dist/ansa-mesh-optimizer" ]; then
        # 显示文件信息
        print_info "文件信息："
        ls -lh dist/ansa-mesh-optimizer
        
        # 测试基本功能
        print_info "测试--version命令..."
        ./dist/ansa-mesh-optimizer --version || true
        
        print_info "测试--help命令..."
        ./dist/ansa-mesh-optimizer --help || true
        
        print_info "测试mock优化器..."
        ./dist/ansa-mesh-optimizer optimize \
            --optimizer genetic \
            --n-calls 3 \
            --evaluator mock \
            --no-display || true
            
        print_success "测试完成"
    else
        print_error "未找到生成的可执行文件"
        exit 1
    fi
}

# 创建发布包
create_release() {
    print_info "创建发布包..."
    
    # 获取版本号
    VERSION="0.2.0"
    if [ -f "VERSION" ]; then
        VERSION=$(cat VERSION | tr -d '[:space:]')
    fi
    
    RELEASE_NAME="ansa-mesh-optimizer-v${VERSION}-linux-x86_64"
    
    # 创建临时目录
    mkdir -p release
    cp dist/ansa-mesh-optimizer release/
    
    # 添加README
    cat > release/README.txt << EOF
ANSA Mesh Optimizer v${VERSION}
==============================

系统要求：
- Linux x86_64 (Ubuntu 20.04+, CentOS 7+, Debian 10+)
- GLIBC >= 2.27

使用方法：
1. 赋予执行权限：chmod +x ansa-mesh-optimizer
2. 查看帮助：./ansa-mesh-optimizer --help
3. 查看版本：./ansa-mesh-optimizer --version

示例命令：
- 使用遗传算法优化：
  ./ansa-mesh-optimizer optimize --optimizer genetic --n-calls 100

- 使用贝叶斯优化：
  ./ansa-mesh-optimizer optimize --optimizer bayesian --n-calls 50

- 使用mock评估器测试：
  ./ansa-mesh-optimizer optimize --optimizer genetic --n-calls 10 --evaluator mock

注意事项：
- 此版本使用NumPy 2.0.2以确保兼容性
- 如遇到库加载问题，请设置：export OPENBLAS_NUM_THREADS=1

构建信息：
- NumPy版本：2.0.2
- PyInstaller版本：6.8.0
- 构建时间：$(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    # 创建tar.gz包
    cd release
    tar -czf "../${RELEASE_NAME}.tar.gz" ansa-mesh-optimizer README.txt
    cd ..
    
    # 生成SHA256校验和
    sha256sum "${RELEASE_NAME}.tar.gz" > "${RELEASE_NAME}.sha256"
    
    print_success "发布包创建成功："
    ls -lh "${RELEASE_NAME}".*
    
    # 显示SHA256
    print_info "SHA256校验和："
    cat "${RELEASE_NAME}.sha256"
}

# 主函数
main() {
    echo "=========================================="
    echo "ANSA Mesh Optimizer Linux打包脚本（兼容版）"
    echo "=========================================="
    echo ""
    
    # 执行各步骤
    check_python
    setup_venv
    install_dependencies
    clean_build
    run_pyinstaller
    test_executable
    create_release
    
    echo ""
    echo "=========================================="
    print_success "打包完成！"
    echo ""
    print_info "可执行文件位置：dist/ansa-mesh-optimizer"
    print_info "发布包位置：ansa-mesh-optimizer-v*.tar.gz"
    echo ""
    print_info "使用方法："
    echo "  1. cd dist"
    echo "  2. ./ansa-mesh-optimizer --help"
    echo "=========================================="
}

# 执行主函数
main