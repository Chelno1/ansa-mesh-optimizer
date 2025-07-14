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
import logging
from pathlib import Path

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Record Python path for debugging
logger.debug("Python path before modification: %s", sys.path)

# 添加当前目录到Python路径
current_dir = str(Path(__file__).parent)
sys.path.insert(0, current_dir)
logger.debug("Added to Python path: %s", current_dir)
logger.debug("Final Python path: %s", sys.path)

def main() -> int:
    """主函数 - 改进的错误处理"""
    try:
        logger.debug("Attempting to import cli_main...")
        from cli.cli_main import main_cli
        logger.debug("Successfully imported cli_main")
        logger.debug("Starting main_cli()")
        return main_cli()
    except ImportError as e:
        print(f"❌ CLI模块导入失败: {e}")
        print("💡 请检查以下可能的问题:")
        print("   - 确保 cli/cli_main.py 文件存在")
        print("   - 检查 Python 路径配置")
        print("   - 验证所有依赖模块已安装")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 130
    except SystemExit as e:
        # 正常的系统退出，传递退出码
        return int(e.code) if e.code is not None else 0
    except Exception as e:
        logger.exception("Uncaught exception in main():")
        print(f"💥 未捕获的异常: {e}")
        print("🔍 如需调试信息，请使用 --debug 参数")
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