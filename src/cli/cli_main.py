#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI主模块 - 命令行接口核心功能
"""

import sys
import argparse
import logging
import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# 全局变量
APP_VERSION = "2.1.0"
APP_NAME = "Ansa Mesh Optimizer"

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建格式化器
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not verbose else detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"警告: 无法创建日志文件 {log_file}: {e}")

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} v{APP_VERSION} - 高级网格参数优化工具',
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
  python main.py optimize --optimizer genetic --evaluator ansa --config my_config.json
        """
    )
    
    # 全局参数
    parser.add_argument('--version', action='version', version=f'{APP_NAME} {APP_VERSION}')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='启用详细输出')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式（仅显示错误）')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 定义要加载的命令模块
    command_modules = [
        ('optimize_cmd', 'register_optimize_command'),
        ('compare_cmd', 'register_compare_command'),
        ('config_cmd', 'register_config_command'),
        ('info_cmd', 'register_info_command'),
        ('test_cmd', 'register_test_command')
    ]
    
    # 安全加载命令模块 - 使用直接导入而不是文件路径
    for module_name, register_func in command_modules:
        try:
            logger.debug("Loading command module: %s", module_name)
            
            # 直接导入模块而不是使用文件路径
            if module_name == 'optimize_cmd':
                from src.cli.commands.optimize_cmd import register_optimize_command as register
            elif module_name == 'compare_cmd':
                from src.cli.commands.compare_cmd import register_compare_command as register
            elif module_name == 'config_cmd':
                from src.cli.commands.config_cmd import register_config_command as register
            elif module_name == 'info_cmd':
                from src.cli.commands.info_cmd import register_info_command as register
            elif module_name == 'test_cmd':
                from src.cli.commands.test_cmd import register_test_command as register
            else:
                logger.error("Unknown command module: %s", module_name)
                continue
            
            # 执行注册函数
            register(subparsers)
            logger.debug("Successfully registered command: %s", module_name)
            
        except Exception as e:
            logger.exception("Failed to register command %s:", module_name)
            print(f"⚠️ 警告: 无法加载命令 {module_name}: {e}")
    
    return parser

def main_cli() -> int:
    """主CLI函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # 设置日志
    setup_logging(args.verbose, args.log_file)
    
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
        print(f"\n⚠️  用户中断程序")
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("程序异常")
        print(f"❌ 程序异常: {e}")
        return 1