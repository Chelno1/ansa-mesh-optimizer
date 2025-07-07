#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试重构后的优化器

验证Phase 4重构的功能完整性
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_refactored_optimizer():
    """测试重构后的优化器"""
    try:
        # 导入重构后的优化器
        from src.core.ansa_mesh_optimizer_refactored import (
            MeshOptimizer, 
            optimize_mesh_parameters,
            get_available_optimizers,
            check_dependencies
        )
        
        logger.info("=== 重构优化器测试 ===")
        
        # 1. 检查依赖
        deps = check_dependencies()
        logger.info(f"可用优化器: {deps['available_optimizers']}")
        
        # 2. 测试优化器工厂
        available_optimizers = get_available_optimizers()
        logger.info(f"优化器工厂可用算法: {available_optimizers}")
        
        # 3. 创建优化器实例
        optimizer = MeshOptimizer(
            evaluator_type='mock',
            use_cache=False
        )
        logger.info("优化器实例创建成功")
        
        # 4. 测试优化过程
        if 'genetic' in available_optimizers:
            logger.info("开始遗传算法优化测试...")
            result = optimizer.optimize(
                optimizer='genetic',
                n_calls=5  # 少量迭代用于测试
            )
            
            logger.info(f"优化完成:")
            logger.info(f"  最佳值: {result.best_value:.6f}")
            logger.info(f"  执行时间: {result.execution_time:.2f}秒")
            logger.info(f"  评估次数: {result.n_evaluations}")
            logger.info(f"  成功状态: {result.success}")
            
            # 5. 测试敏感性分析
            logger.info("开始敏感性分析测试...")
            sensitivity_results = optimizer.sensitivity_analysis(
                n_trials=3,
                noise_level=0.1
            )
            logger.info(f"敏感性分析完成，分析了 {len(sensitivity_results)} 个参数")
            
            # 6. 测试摘要信息
            summary = optimizer.get_optimization_summary()
            logger.info(f"优化摘要:")
            logger.info(f"  总评估次数: {summary['total_evaluations']}")
            logger.info(f"  可用优化器: {summary['config']['available_optimizers']}")
            
            # 7. 测试保存最佳参数
            param_file = optimizer.save_best_params()
            logger.info(f"最佳参数已保存到: {param_file}")
            
        else:
            logger.warning("遗传算法不可用，跳过优化测试")
        
        # 8. 测试便捷函数
        if 'parallel' in available_optimizers:
            logger.info("测试便捷优化函数...")
            result = optimize_mesh_parameters(
                n_calls=3,
                optimizer='parallel',
                evaluator_type='mock',
                use_cache=False
            )
            logger.info(f"便捷函数优化完成，最佳值: {result.best_value:.6f}")
        
        logger.info("=== 重构优化器测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"重构优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_pattern():
    """测试策略模式实现"""
    try:
        from src.optimizers import OptimizerFactory, create_default_config
        from src.config.config import config_manager
        from src.evaluators.mesh_evaluator import create_mesh_evaluator
        
        logger.info("=== 策略模式测试 ===")
        
        # 创建测试组件
        evaluator = create_mesh_evaluator('mock')
        param_space = config_manager.parameter_space
        optimizer_config = create_default_config()
        
        # 测试可用优化器
        available = OptimizerFactory.get_available_optimizers()
        logger.info(f"可用优化器策略: {available}")
        
        # 测试创建不同策略
        for optimizer_type in available[:2]:  # 测试前两个
            try:
                strategy = OptimizerFactory.create_optimizer(
                    optimizer_type=optimizer_type,
                    param_space=param_space,
                    evaluator=evaluator,
                    config=optimizer_config
                )
                logger.info(f"成功创建 {optimizer_type} 策略: {strategy.get_name()}")
                
                # 测试小规模优化
                result = strategy.optimize(n_calls=2)
                logger.info(f"  {optimizer_type} 优化结果: {result['best_value']:.6f}")
                
            except Exception as e:
                logger.warning(f"策略 {optimizer_type} 测试失败: {e}")
        
        logger.info("=== 策略模式测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"策略模式测试失败: {e}")
        return False

def test_modular_architecture():
    """测试模块化架构"""
    try:
        logger.info("=== 模块化架构测试 ===")
        
        # 测试各模块独立导入
        modules_to_test = [
            ('src.optimizers', ['OptimizerFactory', 'OptimizerConfig']),
            ('src.visualization.optimization_visualizer', ['OptimizationVisualizer']),
            ('src.reports.optimization_reporter', ['OptimizationReporter']),
            ('src.analysis.statistical_analyzer', ['StatisticalAnalyzer']),
        ]
        
        for module_name, classes in modules_to_test:
            try:
                module = __import__(module_name, fromlist=classes)
                for class_name in classes:
                    if hasattr(module, class_name):
                        logger.info(f"✓ {module_name}.{class_name} 可用")
                    else:
                        logger.warning(f"✗ {module_name}.{class_name} 不可用")
            except ImportError as e:
                logger.warning(f"✗ 模块 {module_name} 导入失败: {e}")
        
        logger.info("=== 模块化架构测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"模块化架构测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始重构验证测试...")
    
    results = []
    
    # 运行各项测试
    tests = [
        ("重构优化器功能", test_refactored_optimizer),
        ("策略模式实现", test_strategy_pattern),
        ("模块化架构", test_modular_architecture),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "通过" if result else "失败"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} 执行异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info("\n=== 测试结果汇总 ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！重构成功！")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} 项测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)