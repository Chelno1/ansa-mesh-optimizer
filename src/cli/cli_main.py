#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI主模块 - 命令行接口核心功能
"""

import argparse
import importlib
import logging

# 导入统一的日志配置
from src.utils.logging_config import setup_cli_logging

logger = logging.getLogger(__name__)

# 全局变量
APP_VERSION = "2.1.0"
APP_NAME = "Ansa Mesh Optimizer"


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{APP_VERSION} - 高级网格参数优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用贝叶斯优化，20次迭代
  python main.py optimize --optimizer bayesian --n-calls 20 --evaluator mock

  # 无头模式运行（不显示图表窗口）
  python main.py optimize --optimizer bayesian --n-calls 5 --evaluator mock --no-display

  # 比较多个优化器
  python main.py compare --optimizers bayesian random genetic --n-calls 15 --evaluator mock

  # 生成配置文件
  python main.py config generate

  # 检查系统信息
  python main.py info --check-deps

  # 使用真实Ansa评估器
  python main.py optimize --optimizer genetic --evaluator ansa --config src/default_config.json
        """,
    )

    # 全局参数
    parser.add_argument(
        "--version", action="version", version=f"{APP_NAME} {APP_VERSION}"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细输出")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式（仅显示错误）"
    )

    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 定义要加载的命令模块映射 {模块名: 注册函数名}
    command_modules = {
        "optimize_cmd": "register_optimize_command",
        "compare_cmd": "register_compare_command",
        "config_cmd": "register_config_command",
        "info_cmd": "register_info_command",
        "test_cmd": "register_test_command",
    }

    # 使用 importlib.import_module 动态加载命令模块
    for module_name, register_func_name in command_modules.items():
        try:
            logger.debug("动态加载命令模块: %s", module_name)

            # 构建完整的模块路径并动态导入
            full_module_path = f"src.cli.commands.{module_name}"
            module = importlib.import_module(full_module_path)

            # 获取注册函数并调用
            register_func = getattr(module, register_func_name)
            register_func(subparsers)

            logger.debug("成功注册命令模块: %s -> %s", module_name, register_func_name)

        except Exception as e:
            logger.exception("无法注册命令模块 %s:", module_name)
            print(f"⚠️ 警告: 无法加载命令 {module_name}: {e}")

    return parser


def main_cli() -> int:
    """主CLI函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 设置日志级别
    if args.quiet:
        pass
    elif args.verbose:
        pass
    else:
        pass

    # 设置日志
    setup_cli_logging(args.verbose, args.log_file)

    # 检查命令
    if not args.command:
        parser.print_help()
        return 1

    # 导入命令处理器
    try:
        logger.debug("Importing command dispatcher...")
        from src.cli.commands.command_dispatcher import dispatch_command

        logger.debug("Successfully imported command dispatcher")

        logger.debug("Dispatching command: %s", args.command)
        return dispatch_command(args)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断程序")
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("程序异常")
        print(f"❌ 程序异常: {e}")
        return 1
