"""
命令分发器 - 负责将命令分发到对应的处理器
"""

import sys
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_and_import_modules():
    """检查并导入必要模块 - 使用新的依赖管理系统"""
    try:
        # 使用新的统一依赖管理器
        from utils.dependency_manager import dependency_manager
        
        print("🔍 使用统一依赖管理系统检查模块...")
        
        # 获取依赖状态
        status = dependency_manager.get_dependency_status()
        
        # 统计依赖状态
        available_count = sum(1 for s in status.values() if s['available'])
        missing_count = sum(1 for s in status.values() if not s['available'])
        required_missing = sum(1 for s in status.values() if not s['available'] and s['required'])
        
        # 显示检查结果
        print(f"\n📊 依赖检查报告:")
        print(f"   ✓ 可用依赖: {available_count}")
        print(f"   ○ 缺失依赖: {missing_count}")
        print(f"   ❌ 缺失必需依赖: {required_missing}")
        
        # 检查关键模块
        required_modules = [
            'config.config',
            'evaluators.mesh_evaluator',
            'utils.optimization_cache',
            'core.early_stopping',
            'core.genetic_optimizer',
            'utils.utils'
        ]
        
        missing_critical = []
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                missing_critical.append((module_name, str(e)))
        
        if missing_critical:
            print(f"\n❌ 关键模块缺失:")
            for module_name, error in missing_critical:
                print(f"  - {module_name}: {error}")
            return False, missing_critical, list(status.keys())
        
        print(f"\n✅ 所有关键模块已加载")
        return True, [], [name for name, s in status.items() if s['available']]
        
    except ImportError as e:
        print(f"❌ 依赖管理器不可用: {e}")
        # 回退到原始检查方法
        return check_modules_fallback()

def check_modules_fallback():
    """回退的模块检查方法"""
    missing_modules = []
    available_modules = []
    
    # 检查必需的本地模块
    required_local_modules = [
        'config',
        'mesh_evaluator',
        'optimization_cache',
        'early_stopping',
        'genetic_optimizer',
        'utils'
    ]
    
    print("回退检查本地模块...")
    for module_name in required_local_modules:
        try:
            # 构造相对于main.py的导入路径
            if module_name == 'config':
                import_name = 'config.config'
            elif module_name == 'mesh_evaluator':
                import_name = 'evaluators.mesh_evaluator'
            elif module_name == 'optimization_cache':
                import_name = 'utils.optimization_cache'
            elif module_name == 'utils':
                import_name = 'utils.utils'
            else:
                import_name = f'core.{module_name}'
            __import__(import_name)
            available_modules.append(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            missing_modules.append((module_name, str(e)))
            print(f"  ✗ {module_name}: {e}")
    
    if missing_modules:
        print(f"\n❌ 缺少必需模块:")
        for module_name, error in missing_modules:
            print(f"  - {module_name}: {error}")
        return False, missing_modules, available_modules
    
    print(f"\n✓ 所有必需模块已加载")
    return True, [], available_modules

def import_core_modules():
    """导入核心模块 - 使用重构后的配置系统"""
    try:
        from core.ansa_mesh_optimizer import optimize_mesh_parameters, MeshOptimizer, check_dependencies
        from core.compare_optimizers import compare_optimizers
        
        # 使用新的统一配置管理器类（不立即实例化）
        from config.config import UnifiedConfigManager
        
        print("✅ 使用重构后的配置系统")
        return True, (optimize_mesh_parameters, MeshOptimizer, compare_optimizers, UnifiedConfigManager, check_dependencies)
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        # 尝试回退到原始配置系统
        try:
            from core.ansa_mesh_optimizer import optimize_mesh_parameters, MeshOptimizer, check_dependencies
            from core.compare_optimizers import compare_optimizers
            from config.config import config_manager as legacy_config_manager
            print("⚠️  回退到原始配置系统")
            return True, (optimize_mesh_parameters, MeshOptimizer, compare_optimizers, legacy_config_manager, check_dependencies)
        except ImportError as e2:
            print(f"❌ 配置系统完全不可用: {e2}")
            return False, None

def dispatch_command(args) -> int:
    """分发命令到对应的处理器"""
    # 对于info命令，不需要导入复杂模块
    if args.command == 'info':
        from .info_cmd import cmd_info
        return cmd_info(args)
    
    # 检查和导入模块
    print("🔍 检查系统环境...")
    success, missing, available = check_and_import_modules()
    
    if not success:
        print(f"\n❌ 系统环境检查失败")
        print(f"建议操作:")
        print(f"  1. 确保所有必需的Python文件存在")
        print(f"  2. 检查文件权限")
        print(f"  3. 运行: pip install -r requirements.txt")
        return 1
    
    print("✓ 系统环境检查通过")
    
    # 导入核心模块
    print("📦 加载核心模块...")
    success, modules = import_core_modules()
    if not success:
        return 1
    
    print("✓ 核心模块加载成功")
    
    # 分发到对应的命令处理器
    if args.command == 'optimize':
        from .optimize_cmd import cmd_optimize
        return cmd_optimize(args, modules)
    elif args.command == 'compare':
        from .compare_cmd import cmd_compare
        return cmd_compare(args, modules)
    elif args.command == 'config':
        from .config_cmd import cmd_config
        return cmd_config(args, modules)
    elif args.command == 'test':
        from .test_cmd import cmd_test
        return cmd_test(args, modules)
    else:
        print(f"❌ 未知命令: {args.command}")
        return 1