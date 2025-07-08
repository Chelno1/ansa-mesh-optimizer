#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa网格优化器主程序 - 重构版本

作者: Chel
创建日期: 2025-06-19
版本: 1.3.0
更新日期: 2025-07-07
重构: 模块化CLI架构，单一职责原则
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def main() -> int:
    """主函数 - 简化版本，委托给CLI模块"""
    try:
        from cli.cli_main import main_cli
        return main_cli()
    except ImportError as e:
        print(f"❌ CLI模块导入失败: {e}")
        print("请确保所有必需的文件存在")
        return 1
    except Exception as e:
        print(f"💥 未捕获的异常: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 再见!")
        sys.exit(130)
    except Exception as e:
        print(f"💥 未捕获的异常: {e}")
        sys.exit(1)