#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行遗传算法优化器演示和性能基准测试

作者: Chel
创建日期: 2025-07-15
版本: 1.0.0
功能: 演示并行遗传算法的使用和性能对比
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, Any
from pathlib import Path


from src.optimizers.genetic_optimizer import GeneticOptimizer, GeneticConfig
from src.optimizers.parallel_genetic_optimizer import ParallelGeneticOptimizer, ParallelGeneticConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockMeshEvaluator:
    """模拟网格评估器 - 用于性能测试"""
    
    def __init__(self, complexity_factor: float = 1.0):
        """
        初始化模拟评估器
        
        Args:
            complexity_factor: 复杂度因子，控制计算时间
        """
        self.complexity_factor = complexity_factor
        self.evaluation_count = 0
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        模拟网格评估 - 计算复杂的数学函数
        
        Args:
            params: 参数字典
            
        Returns:
            适应度值（越小越好）
        """
        self.evaluation_count += 1
        
        # 模拟计算延迟
        if self.complexity_factor > 0:
            # 执行一些计算密集的操作
            x = np.array(list(params.values()))
            
            # 复杂的多峰函数（Rastrigin函数的变体）
            n = len(x)
            A = 10
            result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
            
            # 添加一些额外的计算来模拟真实的网格评估
            for _ in range(int(self.complexity_factor * 100)):
                temp = np.sin(np.sum(x)) * np.cos(np.prod(x))
                result += temp * 1e-6
            
            return float(result)
        else:
            # 简单的球面函数
            return sum(v**2 for v in params.values())


def create_test_param_space():
    """创建测试参数空间"""
    return {
        'param1': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param2': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param3': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param4': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param5': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param6': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param7': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
        'param8': {'type': 'continuous', 'bounds': [-5.12, 5.12]},
    }


def run_serial_optimization(param_space: Dict, evaluator, n_calls: int = 500) -> Dict[str, Any]:
    """运行串行遗传算法优化"""
    logger.info("开始串行遗传算法优化...")
    
    # 配置串行遗传算法
    config = GeneticConfig(
        population_size=50,
        max_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=5,
        tournament_size=3
    )
    
    # 创建优化器
    optimizer = GeneticOptimizer(param_space, evaluator, config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行优化
    result = optimizer.optimize(n_calls)
    
    # 计算执行时间
    execution_time = time.time() - start_time
    
    return {
        'type': 'serial',
        'result': result,
        'execution_time': execution_time,
        'evaluations': evaluator.evaluation_count
    }


def run_parallel_optimization(param_space: Dict, evaluator, n_calls: int = 500, 
                            n_workers: int = 4) -> Dict[str, Any]:
    """运行并行遗传算法优化"""
    logger.info(f"开始并行遗传算法优化 (工作进程数: {n_workers})...")
    
    # 配置并行遗传算法
    config = ParallelGeneticConfig(
        # 基础遗传算法参数
        population_size=50,
        max_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=5,
        tournament_size=3,
        
        # 并行配置
        n_workers=n_workers,
        use_multiprocessing=True,
        parallel_evaluation=True,
        parallel_diversity=True,
        parallel_evolution=True,
        
        # 批处理配置
        evaluation_batch_size=10,
        evolution_batch_size=20,
        auto_batch_size=True,
        
        # 性能优化
        vectorized_operations=True,
        cache_evaluations=True,
        
        # 性能监控
        performance_monitoring=True,
        log_performance=True
    )
    
    # 创建优化器
    optimizer = ParallelGeneticOptimizer(param_space, evaluator, parallel_config=config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行优化
    result = optimizer.optimize(n_calls)
    
    # 计算执行时间
    execution_time = time.time() - start_time
    
    return {
        'type': 'parallel',
        'result': result,
        'execution_time': execution_time,
        'evaluations': evaluator.evaluation_count,
        'n_workers': n_workers
    }


def compare_performance(param_space: Dict, complexity_factor: float = 1.0, 
                       n_calls: int = 500):
    """比较串行和并行优化的性能"""
    logger.info(f"开始性能比较测试 (复杂度因子: {complexity_factor}, 评估次数: {n_calls})")
    
    results = []
    
    # 测试串行优化
    serial_evaluator = MockMeshEvaluator(complexity_factor)
    serial_result = run_serial_optimization(param_space, serial_evaluator, n_calls)
    results.append(serial_result)
    
    # 测试不同工作进程数的并行优化
    for n_workers in [2, 4, 8]:
        parallel_evaluator = MockMeshEvaluator(complexity_factor)
        parallel_result = run_parallel_optimization(
            param_space, parallel_evaluator, n_calls, n_workers
        )
        results.append(parallel_result)
    
    # 输出性能比较结果
    print("\n" + "="*80)
    print("性能比较结果")
    print("="*80)
    
    serial_time = serial_result['execution_time']
    serial_fitness = serial_result['result'].best_score
    
    print(f"串行优化:")
    print(f"  执行时间: {serial_time:.2f}s")
    print(f"  最佳适应度: {serial_fitness:.6f}")
    print(f"  评估次数: {serial_result['evaluations']}")
    print()
    
    for result in results[1:]:  # 跳过串行结果
        parallel_time = result['execution_time']
        parallel_fitness = result['result'].best_score
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        
        print(f"并行优化 ({result['n_workers']} 工作进程):")
        print(f"  执行时间: {parallel_time:.2f}s")
        print(f"  最佳适应度: {parallel_fitness:.6f}")
        print(f"  评估次数: {result['evaluations']}")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  效率: {speedup/result['n_workers']*100:.1f}%")
        print()
    
    return results


def demonstrate_parallel_features():
    """演示并行遗传算法的特性"""
    logger.info("演示并行遗传算法特性...")
    
    param_space = create_test_param_space()
    evaluator = MockMeshEvaluator(complexity_factor=0.5)
    
    # 创建高级并行配置
    config = ParallelGeneticConfig(
        # 基础参数
        population_size=100,
        max_generations=30,
        
        # 并行配置
        n_workers=4,
        parallel_evaluation=True,
        parallel_diversity=True,
        parallel_evolution=True,
        
        # 高级特性
        vectorized_operations=True,
        cache_evaluations=True,
        auto_batch_size=True,
        
        # 多样性保持
        diversity_preservation=True,
        
        # 性能监控
        performance_monitoring=True,
        memory_monitoring=True,
        
        # 容错机制
        fault_tolerance=True,
        max_retries=3,
        timeout_seconds=60
    )
    
    # 创建优化器
    optimizer = ParallelGeneticOptimizer(param_space, evaluator, parallel_config=config)
    
    print("\n" + "="*80)
    print("并行遗传算法特性演示")
    print("="*80)
    print(f"参数空间维度: {len(param_space)}")
    print(f"种群大小: {config.population_size}")
    print(f"最大代数: {config.max_generations}")
    print(f"工作进程数: {config.n_workers}")
    print(f"并行评估: {config.parallel_evaluation}")
    print(f"并行多样性计算: {config.parallel_diversity}")
    print(f"向量化操作: {config.vectorized_operations}")
    print(f"评估缓存: {config.cache_evaluations}")
    print()
    
    # 执行优化
    start_time = time.time()
    result = optimizer.optimize(n_calls=1000)
    execution_time = time.time() - start_time
    
    print("优化结果:")
    print(f"  最佳适应度: {result.best_score:.6f}")
    print(f"  最佳参数: {result.best_params}")
    print(f"  执行时间: {execution_time:.2f}s")
    print(f"  总代数: {len(result.generation_stats) if result.generation_stats else 0}")
    print(f"  总评估次数: {result.n_evaluations}")
    
    # 获取性能摘要
    if hasattr(optimizer, 'performance_monitor'):
        performance_summary = optimizer.performance_monitor.get_performance_summary()
        print("\n性能摘要:")
        for key, value in performance_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def main():
    """主函数"""
    print("并行遗传算法优化器演示")
    print("="*50)
    
    # 创建测试参数空间
    param_space = create_test_param_space()
    
    try:
        # 1. 演示并行特性
        demonstrate_parallel_features()
        
        # 2. 性能比较测试
        print("\n开始性能比较测试...")
        
        # 轻量级测试
        print("\n轻量级计算测试:")
        compare_performance(param_space, complexity_factor=0.1, n_calls=200)
        
        # 中等复杂度测试
        print("\n中等复杂度测试:")
        compare_performance(param_space, complexity_factor=1.0, n_calls=300)
        
        # 高复杂度测试
        print("\n高复杂度测试:")
        compare_performance(param_space, complexity_factor=2.0, n_calls=400)
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()