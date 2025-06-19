#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa Batch Mesh Optimizer (改进版)

优化有限元网格参数，最小化不合格网格数量

作者: Chel
创建日期: 2025-06-09
版本: 1.1.0
更新日期: 2025-06-19
"""

import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 第三方库导入（带错误处理）
try:
    from skopt import gp_minimize, forest_minimize, dummy_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective
    import matplotlib.pyplot as plt
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize未安装，某些功能可能不可用")

# 本地模块导入
from config import config_manager, OptimizationConfig
from mesh_evaluator import create_mesh_evaluator, MeshEvaluator
from optimization_cache import OptimizationCache, CachedEvaluator
from early_stopping import create_early_stopping, EarlyStopping
from genetic_optimizer_improved import GeneticOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeshOptimizer:
    """网格参数优化器主类"""
    
    def __init__(self, 
                 config: Optional[OptimizationConfig] = None,
                 evaluator_type: str = 'ansa',
                 use_cache: bool = True):
        """
        初始化优化器
        
        Args:
            config: 优化配置对象
            evaluator_type: 评估器类型 ('ansa' 或 'mock')
            use_cache: 是否使用缓存
        """
        self.config = config or config_manager.optimization_config
        self.param_space = config_manager.parameter_space
        
        # 创建评估器
        self.base_evaluator = create_mesh_evaluator(evaluator_type)
        
        # 创建缓存（如果启用）
        if use_cache and self.config.use_cache:
            self.cache = OptimizationCache(self.config.cache_file)
            self.evaluator = CachedEvaluator(self.base_evaluator, self.cache)
        else:
            self.cache = None
            self.evaluator = self.base_evaluator
        
        # 创建早停机制
        if self.config.early_stopping:
            self.early_stopping = create_early_stopping(self.config)
        else:
            self.early_stopping = None
        
        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
    
    def optimize(self, 
                 optimizer: str = 'bayesian',
                 n_calls: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            optimizer: 优化器类型
            n_calls: 优化迭代次数
            **kwargs: 其他优化器参数
            
        Returns:
            优化结果字典
        """
        n_calls = n_calls or self.config.n_calls
        
        logger.info(f"开始使用 {optimizer} 优化器进行网格参数优化")
        logger.info(f"迭代次数: {n_calls}")
        
        start_time = time.time()
        
        try:
            if optimizer.lower() == 'bayesian':
                result = self._optimize_bayesian(n_calls, **kwargs)
            elif optimizer.lower() == 'random':
                result = self._optimize_random(n_calls, **kwargs)
            elif optimizer.lower() == 'forest':
                result = self._optimize_forest(n_calls, **kwargs)
            elif optimizer.lower() in ['genetic', 'ga']:
                result = self._optimize_genetic(n_calls, **kwargs)
            elif optimizer.lower() == 'parallel':
                result = self._optimize_parallel(n_calls, **kwargs)
            else:
                raise ValueError(f"不支持的优化器: {optimizer}")
            
            execution_time = time.time() - start_time
            
            # 完善结果信息
            result.update({
                'optimizer': optimizer,
                'execution_time': execution_time,
                'n_calls': n_calls,
                'config': self.config
            })
            
            self.best_result = result
            
            # 生成报告
            self._generate_optimization_report(result)
            
            logger.info(f"优化完成，执行时间: {execution_time:.2f}秒")
            logger.info(f"最佳目标值: {result['best_value']:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"优化过程中发生错误: {e}")
            raise
        finally:
            # 保存缓存
            if self.cache:
                self.cache._save_cache()
    
    def _optimize_bayesian(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """贝叶斯优化"""
        if not SKOPT_AVAILABLE:
            raise RuntimeError("贝叶斯优化需要安装scikit-optimize")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            result = self._evaluate_with_early_stopping(params)
            return result
        
        result = gp_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result, 'Bayesian Optimization')
    
    def _optimize_random(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """随机搜索优化"""
        if not SKOPT_AVAILABLE:
            raise RuntimeError("随机搜索需要安装scikit-optimize")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_early_stopping(params)
        
        result = dummy_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result, 'Random Search')
    
    def _optimize_forest(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """森林优化"""
        if not SKOPT_AVAILABLE:
            raise RuntimeError("森林优化需要安装scikit-optimize")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_early_stopping(params)
        
        result = forest_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result, 'Forest Optimization')
    
    def _optimize_genetic(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """遗传算法优化"""
        genetic_optimizer = GeneticOptimizer(
            param_space=self.param_space,
            evaluator=self.evaluator,
            config=self.config
        )
        
        return genetic_optimizer.optimize(n_calls, **kwargs)
    
    def _optimize_parallel(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """并行随机搜索"""
        n_workers = kwargs.get('n_workers', mp.cpu_count())
        
        logger.info(f"使用 {n_workers} 个进程进行并行优化")
        
        # 生成随机参数组合
        param_sets = self._generate_random_params(n_calls)
        
        best_value = float('inf')
        best_params = None
        all_results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交任务
            future_to_params = {
                executor.submit(self._evaluate_params_safe, params): params 
                for params in param_sets
            }
            
            # 收集结果
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    all_results.append((params, result))
                    
                    if result < best_value:
                        best_value = result
                        best_params = params
                        
                    logger.info(f"并行评估完成: {result:.6f}")
                    
                except Exception as e:
                    logger.error(f"并行评估失败: {e}")
                    all_results.append((params, float('inf')))
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimizer_name': 'Parallel Random Search',
            'all_results': all_results
        }
    
    def _evaluate_with_early_stopping(self, params: Dict[str, float]) -> float:
        """带早停的评估"""
        result = self.evaluator.evaluate_mesh(params)
        
        # 记录历史
        self.optimization_history.append({
            'params': params.copy(),
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # 检查早停
        if self.early_stopping:
            if self.early_stopping(result, params):
                logger.info("早停触发，停止优化")
                # 这里可以通过异常或其他方式通知优化器停止
        
        return result
    
    def _evaluate_params_safe(self, params: Dict[str, float]) -> float:
        """线程安全的参数评估"""
        try:
            return self.base_evaluator.evaluate_mesh(params)
        except Exception as e:
            logger.error(f"参数评估失败: {e}")
            return float('inf')
    
    def _generate_random_params(self, n_samples: int) -> List[Dict[str, float]]:
        """生成随机参数组合"""
        param_sets = []
        bounds = self.param_space.get_bounds()
        param_names = self.param_space.get_param_names()
        param_types = self.param_space.get_param_types()
        
        np.random.seed(self.config.random_state)
        
        for _ in range(n_samples):
            params = {}
            for i, (name, (low, high), param_type) in enumerate(zip(param_names, bounds, param_types)):
                if param_type == int:
                    params[name] = np.random.randint(low, high + 1)
                else:
                    params[name] = np.random.uniform(low, high)
            param_sets.append(params)
        
        return param_sets
    
    def _format_skopt_result(self, result, optimizer_name: str) -> Dict[str, Any]:
        """格式化scikit-optimize结果"""
        param_names = self.param_space.get_param_names()
        
        best_params = {}
        for i, name in enumerate(param_names):
            best_params[name] = result.x[i]
        
        return {
            'best_params': best_params,
            'best_value': result.fun,
            'optimizer_name': optimizer_name,
            'skopt_result': result
        }
    
    def _generate_optimization_report(self, result: Dict[str, Any]) -> str:
        """生成优化报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer_name = result['optimizer_name'].replace(' ', '_')
        report_dir = Path(f"optimization_reports/{timestamp}_{optimizer_name}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_file = report_dir / "optimization_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"优化报告 - {result['optimizer_name']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"执行时间: {result['execution_time']:.2f} 秒\n")
            f.write(f"迭代次数: {result['n_calls']}\n")
            f.write(f"最佳目标值: {result['best_value']:.6f}\n\n")
            
            f.write("最佳参数:\n")
            for name, value in result['best_params'].items():
                f.write(f"  {name}: {value}\n")
            
            f.write("\n")
            
            # 缓存统计
            if self.cache:
                stats = self.cache.get_stats()
                f.write("缓存统计:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 早停信息
            if self.early_stopping and self.early_stopping.should_stop:
                early_stop_info = self.early_stopping.get_best_result()
                f.write("早停信息:\n")
                for key, value in early_stop_info.items():
                    f.write(f"  {key}: {value}\n")
        
        # 生成可视化图表
        self._generate_plots(result, report_dir)
        
        logger.info(f"详细报告已保存到: {report_dir}")
        return str(report_dir)
    
    def _generate_plots(self, result: Dict[str, Any], report_dir: Path) -> None:
        """生成可视化图表"""
        try:
            # 收敛图
            if 'skopt_result' in result:
                plot_convergence(result['skopt_result'])
                plt.title(f"Convergence - {result['optimizer_name']}")
                plt.savefig(report_dir / "convergence.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 早停历史图
            if self.early_stopping:
                self.early_stopping.plot_history(report_dir / "early_stopping_history.png")
            
            # 参数相关性图（如果数据足够）
            if 'skopt_result' in result and result['n_calls'] >= 20:
                try:
                    plot_objective(result['skopt_result'])
                    plt.savefig(report_dir / "parameter_correlation.png", dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"无法生成参数相关性图: {e}")
            
        except Exception as e:
            logger.warning(f"生成图表失败: {e}")
    
    def sensitivity_analysis(self, 
                           best_params: Optional[Dict[str, float]] = None,
                           n_trials: int = 5,
                           noise_level: float = 0.1) -> Dict[str, List]:
        """
        参数敏感性分析
        
        Args:
            best_params: 最佳参数（如果为None则使用最近的优化结果）
            n_trials: 每个参数的试验次数
            noise_level: 参数扰动幅度
            
        Returns:
            敏感性分析结果
        """
        if best_params is None:
            if self.best_result is None:
                raise ValueError("没有可用的最佳参数，请先运行优化或提供参数")
            best_params = self.best_result['best_params']
        
        logger.info("开始参数敏感性分析...")
        
        sensitivity_results = {}
        bounds = self.param_space.get_bounds()
        param_names = self.param_space.get_param_names()
        param_types = self.param_space.get_param_types()
        
        for i, param_name in enumerate(param_names):
            param_value = best_params[param_name]
            param_type = param_types[i]
            low, high = bounds[i]
            
            logger.info(f"分析参数: {param_name}")
            
            # 确定参数类型并设置合适的扰动范围
            if param_type == float:
                min_val = max(low, param_value * (1 - noise_level))
                max_val = min(high, param_value * (1 + noise_level))
                test_values = np.linspace(min_val, max_val, n_trials)
            else:  # int
                min_val = max(low, int(param_value - param_value * noise_level))
                max_val = min(high, int(param_value + param_value * noise_level))
                test_values = np.linspace(min_val, max_val, n_trials, dtype=int)
            
            # 测试不同参数值的影响
            results = []
            for test_val in test_values:
                test_params = best_params.copy()
                test_params[param_name] = test_val
                
                result = self.evaluator.evaluate_mesh(test_params)
                results.append((test_val, result))
                logger.debug(f"  {param_name}={test_val:.4f} -> {result:.4f}")
            
            sensitivity_results[param_name] = results
        
        # 生成敏感性分析图表
        self._plot_sensitivity_analysis(sensitivity_results, best_params)
        
        logger.info("参数敏感性分析完成")
        return sensitivity_results
    
    def _plot_sensitivity_analysis(self, 
                                  sensitivity_results: Dict[str, List],
                                  best_params: Dict[str, float]) -> None:
        """绘制敏感性分析图表"""
        try:
            n_params = len(sensitivity_results)
            fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
            
            if n_params == 1:
                axes = [axes]
            
            for i, (param_name, results) in enumerate(sensitivity_results.items()):
                test_values, objectives = zip(*results)
                axes[i].plot(test_values, objectives, 'o-', linewidth=2, markersize=6)
                axes[i].axvline(x=best_params[param_name], color='r', linestyle='--', linewidth=2)
                axes[i].set_title(f'敏感性分析: {param_name}')
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('目标值')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sensitivity_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"敏感性分析图表已保存: {filename}")
            
        except Exception as e:
            logger.warning(f"生成敏感性分析图表失败: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要信息"""
        summary = {
            'total_evaluations': len(self.optimization_history),
            'best_result': self.best_result,
            'config': {
                'optimizer': self.config.optimizer,
                'n_calls': self.config.n_calls,
                'early_stopping': self.config.early_stopping,
                'use_cache': self.config.use_cache
            }
        }
        
        if self.cache:
            summary['cache_stats'] = self.cache.get_stats()
        
        if self.early_stopping:
            summary['early_stopping_info'] = self.early_stopping.get_best_result()
        
        return summary
    
    def save_best_params(self, filename: Optional[str] = None) -> str:
        """
        保存最佳参数到文件
        
        Args:
            filename: 保存文件名（可选）
            
        Returns:
            保存的文件路径
        """
        if self.best_result is None:
            raise ValueError("没有可用的最佳参数，请先运行优化")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            optimizer_name = self.best_result['optimizer_name'].replace(' ', '_')
            filename = f"best_params_{optimizer_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 最佳网格参数 - {self.best_result['optimizer_name']}\n")
            f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
            f.write(f"# 最佳目标值: {self.best_result['best_value']:.6f}\n\n")
            
            for key, value in self.best_result['best_params'].items():
                f.write(f"{key} = {value}\n")
        
        logger.info(f"最佳参数已保存到: {filename}")
        return filename

def optimize_mesh_parameters(
    n_calls: int = 20,
    optimizer: str = 'bayesian',
    evaluator_type: str = 'ansa',
    config_file: Optional[str] = None,
    use_cache: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷的网格参数优化函数
    
    Args:
        n_calls: 优化迭代次数
        optimizer: 优化器类型
        evaluator_type: 评估器类型
        config_file: 配置文件路径
        use_cache: 是否使用缓存
        **kwargs: 其他优化器参数
        
    Returns:
        优化结果字典
    """
    # 加载配置
    if config_file:
        config_manager.load_config(config_file)
    
    # 创建优化器
    mesh_optimizer = MeshOptimizer(
        evaluator_type=evaluator_type,
        use_cache=use_cache
    )
    
    # 执行优化
    result = mesh_optimizer.optimize(
        optimizer=optimizer,
        n_calls=n_calls,
        **kwargs
    )
    
    # 运行敏感性分析（如果启用）
    if config_manager.optimization_config.sensitivity_analysis:
        try:
            mesh_optimizer.sensitivity_analysis(
                n_trials=config_manager.optimization_config.sensitivity_trials,
                noise_level=config_manager.optimization_config.noise_level
            )
        except Exception as e:
            logger.warning(f"敏感性分析失败: {e}")
    
    # 保存最佳参数
    mesh_optimizer.save_best_params()
    
    return result

def compare_optimizers(
    optimizers: List[str] = ['bayesian', 'random', 'forest', 'genetic'],
    n_calls: int = 30,
    evaluator_type: str = 'mock',  # 默认使用模拟评估器以便快速比较
    **kwargs
) -> Dict[str, Any]:
    """
    比较不同优化器的性能
    
    Args:
        optimizers: 要比较的优化器列表
        n_calls: 每个优化器的迭代次数
        evaluator_type: 评估器类型
        **kwargs: 其他参数
        
    Returns:
        比较结果字典
    """
    logger.info(f"开始比较 {len(optimizers)} 个优化器")
    
    results = {}
    
    for optimizer in optimizers:
        logger.info(f"测试优化器: {optimizer}")
        
        try:
            start_time = time.time()
            
            result = optimize_mesh_parameters(
                n_calls=n_calls,
                optimizer=optimizer,
                evaluator_type=evaluator_type,
                use_cache=False,  # 比较时不使用缓存
                **kwargs
            )
            
            execution_time = time.time() - start_time
            
            results[optimizer] = {
                'best_value': result['best_value'],
                'best_params': result['best_params'],
                'execution_time': execution_time,
                'optimizer_name': result['optimizer_name']
            }
            
            logger.info(f"{optimizer}: 最佳值={result['best_value']:.6f}, 时间={execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"优化器 {optimizer} 执行失败: {e}")
            results[optimizer] = {
                'error': str(e),
                'best_value': float('inf'),
                'execution_time': 0
            }
    
    # 生成比较报告
    _generate_comparison_report(results)
    
    return results

def _generate_comparison_report(results: Dict[str, Dict]) -> None:
    """生成优化器比较报告"""
    try:
        import pandas as pd
        
        # 创建DataFrame
        data = []
        for optimizer, result in results.items():
            if 'error' not in result:
                data.append({
                    'optimizer': optimizer,
                    'best_value': result['best_value'],
                    'execution_time': result['execution_time']
                })
        
        if not data:
            logger.warning("没有成功的优化结果，无法生成比较报告")
            return
        
        df = pd.DataFrame(data)
        
        # 保存CSV报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"optimizer_comparison_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # 生成可视化图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 目标值比较
        ax1.bar(df['optimizer'], df['best_value'])
        ax1.set_title('最佳目标值比较')
        ax1.set_xlabel('优化器')
        ax1.set_ylabel('目标值')
        ax1.tick_params(axis='x', rotation=45)
        
        # 执行时间比较
        ax2.bar(df['optimizer'], df['execution_time'])
        ax2.set_title('执行时间比较')
        ax2.set_xlabel('优化器')
        ax2.set_ylabel('时间 (秒)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f"optimizer_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"比较报告已保存: {csv_file}, {plot_file}")
        
    except ImportError:
        logger.warning("pandas未安装，无法生成详细比较报告")
    except Exception as e:
        logger.error(f"生成比较报告失败: {e}")

# 向后兼容的函数别名
def optimize_mesh_parameters_legacy(*args, **kwargs):
    """向后兼容的优化函数"""
    warnings.warn("optimize_mesh_parameters_legacy已弃用，请使用optimize_mesh_parameters", 
                  DeprecationWarning, stacklevel=2)
    return optimize_mesh_parameters(*args, **kwargs)

if __name__ == "__main__":
    # 示例用法
    logger.info("Ansa网格优化器示例")
    
    # 单个优化器测试
    result = optimize_mesh_parameters(
        n_calls=20,
        optimizer='mock',  # 使用模拟评估器进行快速测试
        evaluator_type='mock'
    )
    
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳值: {result['best_value']:.6f}")
    
    # 多优化器比较
    comparison_results = compare_optimizers(
        optimizers=['random', 'genetic'],
        n_calls=10,
        evaluator_type='mock'
    )
    
    print("\n优化器比较结果:")
    for optimizer, result in comparison_results.items():
        if 'error' not in result:
            print(f"{optimizer}: {result['best_value']:.6f}")