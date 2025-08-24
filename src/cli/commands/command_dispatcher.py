"""
命令分发器 - 负责将命令分发到对应的处理器
"""

import importlib
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def check_and_import_modules() -> Tuple[bool, List[Tuple[str, str]], List[str]]:
    """检查并导入必要模块 - 使用新的依赖管理系统"""
    try:
        logger.debug("Importing dependency manager...")
        from ...utils.dependency_manager import dependency_manager

        logger.info("🔍 使用统一依赖管理系统检查模块...")

        # 获取依赖状态
        logger.debug("Getting dependency status...")
        status = dependency_manager.get_dependency_status()

        # 统计依赖状态
        available_count = sum(1 for s in status.values() if s["available"])
        missing_count = sum(1 for s in status.values() if not s["available"])
        required_missing = sum(
            1 for s in status.values() if not s["available"] and s["required"]
        )

        # 显示检查结果
        print("\n📊 依赖检查报告:")
        print(f"   ✓ 可用依赖: {available_count}")
        print(f"   ○ 缺失依赖: {missing_count}")
        print(f"   ❌ 缺失必需依赖: {required_missing}")

        # 检查关键模块
        required_modules = [
            "config.config",
            "evaluators.mesh_evaluator",
            "utils.optimization_cache",
            "core.early_stopping",
            "optimizers.genetic_optimizer",
            "utils.utils",
        ]

        missing_critical = []
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                missing_critical.append((module_name, str(e)))

        if missing_critical:
            print("\n❌ 关键模块缺失:")
            for module_name, error in missing_critical:
                print(f"  - {module_name}: {error}")
            return False, missing_critical, list(status.keys())

        print("\n✅ 所有关键模块已加载")
        return True, [], [name for name, s in status.items() if s["available"]]

    except ImportError as e:
        logger.exception("Dependency manager not available")
        print(f"❌ 依赖管理器不可用: {e}")
        # 回退到基本检查
        return False, [("dependency_manager", str(e))], []
    except Exception as e:
        logger.exception("Unexpected error in module check")
        print(f"❌ 模块检查异常: {e}")
        return False, [("module_check", str(e))], []


def import_core_modules():
    """导入核心模块 - 使用重构后的配置系统"""
    try:
        # 使用新的统一配置管理器类
        from ...config.config import UnifiedConfigManager
        from ...core.ansa_mesh_optimizer import (
            MeshOptimizer,
            check_dependencies,
            optimize_mesh_parameters,
        )
        from ...core.compare_optimizers import compare_optimizers

        print("✅ 使用重构后的配置系统")
        return True, (
            optimize_mesh_parameters,
            MeshOptimizer,
            compare_optimizers,
            UnifiedConfigManager,
            check_dependencies,
        )
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        return False, None


def dispatch_command(args) -> int:
    """分发命令到对应的处理器"""
    logger.debug("Dispatching command: %s", args.command)

    # 对于info命令，不需要导入复杂模块
    if args.command == "info":
        try:
            logger.debug("Importing info command handler...")
            from .info_cmd import cmd_info

            return cmd_info(args)
        except ImportError as e:
            logger.exception("Failed to import info command handler")
            print(f"❌ 无法加载info命令: {e}")
            return 1

    # 检查和导入模块
    logger.info("🔍 检查系统环境...")
    try:
        success, missing, available = check_and_import_modules()
        logger.debug(
            "Module check result - Success: %s, Missing: %s, Available: %s",
            success,
            missing,
            available,
        )

        if not success:
            print("\n❌ 系统环境检查失败")
            print("建议操作:")
            print("  1. 确保所有必需的Python文件存在")
            print("  2. 检查文件权限")
            print("  3. 运行: pip install -r requirements.txt")
            return 1

        print("✓ 系统环境检查通过")

        # 导入核心模块
        print("📦 加载核心模块...")
        success, modules = import_core_modules()
        if not success:
            return 1

        print("✓ 核心模块加载成功")

    except Exception as e:
        logger.exception("System environment check failed")
        print(f"❌ 系统环境检查异常: {e}")
        return 1

    # 分发到对应的命令处理器
    command_handlers = {
        "optimize": (".optimize_cmd", "cmd_optimize"),
        "compare": (".compare_cmd", "cmd_compare"),
        "config": (".config_cmd", "cmd_config"),
        "test": (".test_cmd", "cmd_test"),
    }

    if args.command not in command_handlers:
        logger.error("Unknown command: %s", args.command)
        print(f"❌ 未知命令: {args.command}")
        return 1

    try:
        module_path, handler_name = command_handlers[args.command]
        logger.debug("Importing command handler: %s from %s", handler_name, module_path)

        module = importlib.import_module(module_path, package=__package__)
        handler = getattr(module, handler_name)

        logger.debug("Executing command handler: %s", handler_name)
        return handler(args, modules)

    except ImportError as e:
        logger.exception("Failed to import command handler")
        print(f"❌ 无法加载命令处理器: {e}")
        return 1
    except AttributeError as e:
        logger.exception("Command handler not found")
        print(f"❌ 命令处理器未找到: {e}")
        return 1
    except Exception as e:
        logger.exception("Command execution failed")
        print(f"❌ 命令执行失败: {e}")
        return 1
