#!/bin/bash
# ANSA Mesh Optimizer 快速启动脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OPTIMIZER="$SCRIPT_DIR/dist/ansa-mesh-optimizer"

if [ ! -f "$OPTIMIZER" ]; then
    echo "错误: 找不到优化器可执行文件"
    echo "请先运行 build_linux_fixed.sh 进行打包"
    exit 1
fi

# 运行优化器，传递所有参数
exec "$OPTIMIZER" "$@"
