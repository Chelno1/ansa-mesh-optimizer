#!/usr/bin/env python3
"""
并行遗传算法使用示例

展示如何在ANSA网格优化项目中使用新的并行遗传算法优化器
"""

import sys
import os
import logging
from pathlib import Path


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数 - 演示并行遗传算法的使用"""
    
    print("=" * 70)
    print("ANSA网格优化 - 并行遗传算法使用示例")
    print("=" * 70)
    
    try:
        # 导入必要的模块
        from src.optimizers.optimizer_strategies import OptimizerFactory
        from src.optimizers.optimizer_config import OptimizerConfig, create_parallel_config
        
        print("✅ 成功导入优化器模块")
        
        # 1. 检查可用的优化器
        print("\n1. 检查可用优化器:")
        available_optimizers = OptimizerFactory.get_available_optimizers()
        print(f"   可用优化器: {available_optimizers}")
        
        if 'parallel_genetic' not in available_optimizers:
            print("❌ 并行遗传算法不可用")
            return
        
        print("✅ 并行遗传算法可用")
        
        # 2. 创建模拟的参数空间和评估器
        print("\n2. 创建测试环境:")
        
        class MockParameterSpace:
            def __init__(self):
                self.params = {
                    'element_size': {'type': float, 'bounds': (0.1, 2.0)},
                    'aspect_ratio': {'type': float, 'bounds': (1.0, 5.0)},
                    'skewness': {'type': float, 'bounds': (0.1, 0.9)},
                    'taper': {'type': float, 'bounds': (0.1, 0.8)}
                }
            
            def get_param_names(self):
                return list(self.params.keys())
            
            def get_parameter_names(self):
                return list(self.params.keys())
            
            def get_bounds(self):
                return [self.params[name]['bounds'] for name in self.get_param_names()]
            
            def get_param_types(self):
                return [self.params[name]['type'] for name in self.get_param_names()]
            
            def get_parameter_types(self):
                return [self.params[name]['type'] for name in self.get_param_names()]
        
        class MockMeshEvaluator:
            def evaluate_mesh(self, params):
                """模拟网格质量评估 - 目标是最小化不合格网格数量"""
                element_size = params.get('element_size', 1.0)
                aspect_ratio = params.get('aspect_ratio', 2.0)
                skewness = params.get('skewness', 0.5)
                taper = params.get('taper', 0.5)
                
                # 模拟网格质量评估函数
                # 理想值: element_size=0.5, aspect_ratio=1.5, skewness=0.2, taper=0.3
                quality_score = (
                    (element_size - 0.5)**2 * 10 +
                    (aspect_ratio - 1.5)**2 * 5 +
                    (skewness - 0.2)**2 * 20 +
                    (taper - 0.3)**2 * 15
                )
                
                # 添加一些噪声来模拟真实评估的变化
                import random
                noise = random.uniform(-0.1, 0.1)
                
                return quality_score + noise
        
        param_space = MockParameterSpace()
        evaluator = MockMeshEvaluator()
        
        print(f"   参数空间: {param_space.get_param_names()}")
        print(f"   目标: 最小化网格质量评分")
        
        # 3. 创建并行遗传算法配置
        print("\n3. 创建并行遗传算法配置:")
        
        # 使用预定义的并行配置
        config = create_parallel_config(n_workers=4)
        
        # 自定义遗传算法参数
        config.update(
            population_size=40,
            n_generations=20,
            crossover_rate=0.8,
            mutation_rate=0.15,
            tournament_size=3,
            elitism_ratio=0.1
        )
        
        print(f"   种群大小: {config.population_size}")
        print(f"   代数: {config.n_generations}")
        print(f"   工作进程数: {config.n_workers}")
        print(f"   交叉率: {config.crossover_rate}")
        print(f"   变异率: {config.mutation_rate}")
        
        # 4. 创建并行遗传算法优化器
        print("\n4. 创建并行遗传算法优化器:")
        
        optimizer = OptimizerFactory.create_optimizer(
            optimizer_type='parallel_genetic',
            param_space=param_space,
            evaluator=evaluator,
            config=config
        )
        
        print(f"✅ 成功创建优化器: {optimizer.get_name()}")
        
        # 5. 执行优化
        print("\n5. 执行并行遗传算法优化:")
        print("   开始优化...")
        
        result = optimizer.optimize(n_calls=100)
        
        print("✅ 优化完成!")
        
        # 6. 分析结果
        print("\n6. 优化结果分析:")
        print(f"   最佳网格质量评分: {result.best_value:.6f}")
        print(f"   最佳参数:")
        for param_name, value in result.best_params.items():
            print(f"     {param_name}: {value:.4f}")
        
        print(f"   优化器名称: {result.optimizer_name}")
        print(f"   总评估次数: {result.n_evaluations}")
        
        # 检查并行特定信息
        if hasattr(result, '_dict_data') and 'parallel_config' in result._dict_data:
            parallel_info = result._dict_data['parallel_config']
            print(f"   并行配置:")
            print(f"     工作进程数: {parallel_info.get('n_workers', 'N/A')}")
            print(f"     并行评估: {parallel_info.get('parallel_evaluation', 'N/A')}")
            print(f"     并行多样性: {parallel_info.get('parallel_diversity', 'N/A')}")
            print(f"     向量化操作: {parallel_info.get('vectorized_operations', 'N/A')}")
        
        # 检查性能摘要
        if hasattr(result, '_dict_data') and 'performance_summary' in result._dict_data:
            perf_summary = result._dict_data['performance_summary']
            print(f"   性能摘要:")
            print(f"     总运行时间: {perf_summary.get('total_runtime', 'N/A'):.2f}秒")
            print(f"     平均评估时间: {perf_summary.get('avg_evaluation_time', 'N/A'):.4f}秒")
            print(f"     平均进化时间: {perf_summary.get('avg_evolution_time', 'N/A'):.4f}秒")
            print(f"     平均多样性计算时间: {perf_summary.get('avg_diversity_time', 'N/A'):.4f}秒")
        
        # 7. 与理想值比较
        print("\n7. 与理想参数比较:")
        ideal_params = {
            'element_size': 0.5,
            'aspect_ratio': 1.5,
            'skewness': 0.2,
            'taper': 0.3
        }
        
        print("   理想参数 vs 优化结果:")
        for param_name in ideal_params:
            ideal_val = ideal_params[param_name]
            optimized_val = result.best_params.get(param_name, 0)
            diff = abs(ideal_val - optimized_val)
            print(f"     {param_name}: 理想={ideal_val:.3f}, 优化={optimized_val:.4f}, 差异={diff:.4f}")
        
        # 8. 使用建议
        print("\n8. 实际使用建议:")
        print("   在真实的ANSA网格优化中:")
        print("   - 将MockMeshEvaluator替换为真实的ANSA评估器")
        print("   - 根据网格复杂度调整种群大小和代数")
        print("   - 根据计算资源调整工作进程数")
        print("   - 使用缓存来避免重复评估")
        print("   - 启用早停机制来节省计算时间")
        
        print("\n" + "=" * 70)
        print("✅ 并行遗传算法使用示例完成!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保项目路径正确且所有依赖已安装")
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_different_configurations():
    """演示不同的配置选项"""
    
    print("\n" + "=" * 70)
    print("不同配置选项演示")
    print("=" * 70)
    
    try:
        from src.optimizers.optimizer_config import OptimizerConfig
        
        # 快速配置 - 适合快速测试
        print("\n1. 快速配置 (适合快速测试):")
        fast_config = OptimizerConfig(
            population_size=20,
            n_generations=10,
            n_workers=2,
            early_stopping=True,
            max_stagnation_iterations=5
        )
        print(f"   种群大小: {fast_config.population_size}")
        print(f"   代数: {fast_config.n_generations}")
        print(f"   工作进程: {fast_config.n_workers}")
        print(f"   早停: {fast_config.early_stopping}")
        
        # 平衡配置 - 适合一般使用
        print("\n2. 平衡配置 (适合一般使用):")
        balanced_config = OptimizerConfig(
            population_size=50,
            n_generations=50,
            n_workers=4,
            early_stopping=True,
            max_stagnation_iterations=15
        )
        print(f"   种群大小: {balanced_config.population_size}")
        print(f"   代数: {balanced_config.n_generations}")
        print(f"   工作进程: {balanced_config.n_workers}")
        print(f"   早停: {balanced_config.early_stopping}")
        
        # 高精度配置 - 适合重要优化
        print("\n3. 高精度配置 (适合重要优化):")
        precision_config = OptimizerConfig(
            population_size=100,
            n_generations=100,
            n_workers=8,
            early_stopping=False,
            convergence_threshold=1e-8
        )
        print(f"   种群大小: {precision_config.population_size}")
        print(f"   代数: {precision_config.n_generations}")
        print(f"   工作进程: {precision_config.n_workers}")
        print(f"   早停: {precision_config.early_stopping}")
        print(f"   收敛阈值: {precision_config.convergence_threshold}")
        
    except ImportError as e:
        print(f"❌ 配置演示失败: {e}")

if __name__ == "__main__":
    main()
    demonstrate_different_configurations()