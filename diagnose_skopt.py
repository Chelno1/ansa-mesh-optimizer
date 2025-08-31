#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断 scikit-optimize 依赖检测问题
"""

print("=== scikit-optimize 依赖诊断 ===\n")

# 1. 直接导入测试
print("1. 直接导入测试:")
try:
    import skopt
    print(f"   ✓ 直接导入 skopt 成功，版本: {skopt.__version__}")
except ImportError as e:
    print(f"   ❌ 直接导入 skopt 失败: {e}")

# 2. dependency_manager 检测
print("\n2. dependency_manager 检测:")
try:
    from src.utils.dependency_manager import dependency_manager
    print(f"   scikit-optimize 可用性: {dependency_manager.is_available('scikit-optimize')}")
    print(f"   错误信息: {dependency_manager.errors.get('scikit-optimize', '无')}")
except Exception as e:
    print(f"   ❌ dependency_manager 检测失败: {e}")

# 3. optimizer_strategies 中的 SKOPT_MODULES 检测
print("\n3. optimizer_strategies SKOPT_MODULES 检测:")
try:
    from src.optimizers.optimizer_strategies import SKOPT_MODULES
    print(f"   SKOPT_MODULES['available']: {SKOPT_MODULES['available']}")
    print(f"   SKOPT_MODULES 内容: {list(SKOPT_MODULES.keys())}")
    if not SKOPT_MODULES['available']:
        print("   详细检查 safe_import_skopt():")
        from src.optimizers.optimizer_strategies import safe_import_skopt
        result = safe_import_skopt()
        print(f"   safe_import_skopt() 返回: {result}")
except Exception as e:
    print(f"   ❌ SKOPT_MODULES 检测失败: {e}")

# 4. 核心优化器的依赖检查
print("\n4. 核心优化器的依赖检查:")
try:
    from src.core.ansa_mesh_optimizer import check_dependencies
    deps = check_dependencies()
    print(f"   check_dependencies() 结果: {deps}")
except Exception as e:
    print(f"   ❌ check_dependencies() 失败: {e}")

# 5. 手动测试 skopt 具体模块导入
print("\n5. 手动测试 skopt 具体模块导入:")
modules_to_test = ['gp_minimize', 'forest_minimize', 'dummy_minimize', 'use_named_args']
for module_name in modules_to_test:
    try:
        module = getattr(skopt, module_name, None)
        if module:
            print(f"   ✓ skopt.{module_name} 可用")
        else:
            # 尝试从子模块导入
            if module_name == 'use_named_args':
                from skopt.utils import use_named_args
                print(f"   ✓ skopt.utils.{module_name} 可用")
            else:
                print(f"   ❌ skopt.{module_name} 不可用")
    except Exception as e:
        print(f"   ❌ skopt.{module_name} 导入失败: {e}")

print("\n=== 诊断完成 ===")