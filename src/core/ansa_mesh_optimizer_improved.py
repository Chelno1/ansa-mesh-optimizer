#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa Batch Mesh Optimizer (改进版本)

优化有限元网格参数，最小化不合格网格数量

作者: Chel
创建日期: 2025-06-09
版本: 1.2.0
更新日期: 2025-06-20
修复: 导入处理，参数验证，错误处理，内存优化，matplotlib显示配置
"""

import numpy as np
import logging
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 安全导入可选依赖
def safe_import_optional_modules():
    """安全导入可选模块"""
    modules = {}
    modules['available'] = []
    
    # scikit-optimize
    try:
        from skopt import gp_minimize, forest_minimize, dummy_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        from skopt.plots import plot_convergence, plot_objective
        
        # 导入显示配置
        from utils.display_config import configure_matplotlib_for_display, safe_show, safe_close
        configure_matplotlib_for_display()
        import matplotlib.pyplot as plt
        
        modules['skopt_available'] = True
        modules['gp_minimize'] = gp_minimize
        modules['forest_minimize'] = forest_minimize
        modules['dummy_minimize'] = dummy_minimize
        modules['Real'] = Real
        modules['Integer'] = Integer
        modules['use_named_args'] = use_named_args
        modules['plot_convergence'] = plot_convergence
        modules['plot_objective'] = plot_objective
        modules['plt'] = plt
        modules['safe_show'] = safe_show
        modules['safe_close'] = safe_close
        modules['available'].append('scikit-optimize')
        logger.info("scikit-optimize 模块加载成功")
        
    except ImportError as e:
        modules['skopt_available'] = False
        modules['skopt_error'] = str(e)
        logger.warning(f"scikit-optimize不可用: {e}")
    
    # pandas (用于报告生成)
    try:
        import pandas as pd
        modules['pandas_available'] = True
        modules['pd'] = pd
        modules['available'].append('pandas')
    except ImportError as e:
        modules['pandas_available'] = False
        modules['pandas_error'] = str(e)
        logger.warning(f"pandas不可用: {e}")
    
    return modules

# 尝试导入字体配置模块
try:
    from utils.font_decorator import with_chinese_font
    DECORATOR_AVAILABLE = True
except ImportError:
    DECORATOR_AVAILABLE = False
    def with_chinese_font(func):
        return func

# 全局模块状态
OPTIONAL_MODULES = safe_import_optional_modules()

# 本地模块导入
try:
    from config.config import config_manager, OptimizationConfig
    from evaluators.mesh_evaluator import create_mesh_evaluator, MeshEvaluator
    from utils.optimization_cache import OptimizationCache, CachedEvaluator
    from core.early_stopping import create_early_stopping, EarlyStopping
    from core.genetic_optimizer_improved import GeneticOptimizer
    from utils.utils import normalize_params, validate_param_types, performance_monitor
except ImportError as e:
    logger.error(f"本地模块导入失败: {e}")
    logger.error("请确保所有必需的模块文件存在")
    raise

class MeshOptimizer:
    """网格参数优化器主类 - 改进版本"""
    
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
        
        # 验证配置
        is_valid, error_msg = self.config.validate()
        if not is_valid:
            raise ValueError(f"配置验证失败: {error_msg}")
        
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
        
        logger.info(f"优化器初始化完成 - 评估器: {evaluator_type}, 缓存: {use_cache}")
    
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
        
        # 验证优化器可用性
        if not self._check_optimizer_availability(optimizer):
            raise ValueError(f"优化器 {optimizer} 不可用或缺少依赖")
        
        with performance_monitor(f"{optimizer} 优化"):
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
                
                # 完善结果信息
                result.update({
                    'optimizer': optimizer,
                    'n_calls': n_calls,
                    'config': self.config,
                    'total_evaluations': len(self.optimization_history)
                })
                
                self.best_result = result
                
                # 生成报告
                try:
                    report_dir = self._generate_optimization_report(result)
                    result['report_dir'] = report_dir
                except Exception as e:
                    logger.warning(f"报告生成失败: {e}")
                
                logger.info(f"优化完成")
                logger.info(f"最佳目标值: {result['best_value']:.6f}")
                
                return result
                
            except Exception as e:
                logger.error(f"优化过程中发生错误: {e}")
                raise
            finally:
                # 保存缓存
                if self.cache:
                    try:
                        self.cache._save_cache()
                    except Exception as e:
                        logger.warning(f"缓存保存失败: {e}")
    
    def _check_optimizer_availability(self, optimizer: str) -> bool:
        """检查优化器可用性"""
        if optimizer.lower() in ['bayesian', 'random', 'forest']:
            if not OPTIONAL_MODULES.get('skopt_available', False):
                logger.error(f"优化器 {optimizer} 需要 scikit-optimize 库")
                return False
        return True
    
    def _optimize_bayesian(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """贝叶斯优化"""
        if not OPTIONAL_MODULES.get('skopt_available', False):
            raise RuntimeError("贝叶斯优化需要安装scikit-optimize")
        
        @OPTIONAL_MODULES['use_named_args'](self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_early_stopping(params)
        
        result = OPTIONAL_MODULES['gp_minimize'](
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
        if not OPTIONAL_MODULES.get('skopt_available', False):
            raise RuntimeError("随机搜索需要安装scikit-optimize")
        
        @OPTIONAL_MODULES['use_named_args'](self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_early_stopping(params)
        
        result = OPTIONAL_MODULES['dummy_minimize'](
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
        if not OPTIONAL_MODULES.get('skopt_available', False):
            raise RuntimeError("森林优化需要安装scikit-optimize")
        
        @OPTIONAL_MODULES['use_named_args'](self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_early_stopping(params)
        
        result = OPTIONAL_MODULES['forest_minimize'](
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
        n_workers = kwargs.get('n_workers', min(mp.cpu_count(), 4))
        
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
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    all_results.append((params, result))
                    
                    if result < best_value:
                        best_value = result
                        best_params = params
                    
                    completed += 1
                    if completed % 5 == 0:
                        logger.info(f"并行评估进度: {completed}/{n_calls}")
                    
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
        try:
            # 标准化参数
            normalized_params = normalize_params(params)
            
            # 验证参数类型
            validated_params = validate_param_types(normalized_params, self.param_space)
            
            result = self.evaluator.evaluate_mesh(validated_params)
            
            # 确保返回有效的浮点数
            if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid evaluation result: {result}")
                return float('inf')
            
            result_float = float(result)
            
            # 记录历史
            self.optimization_history.append({
                'params': validated_params.copy(),
                'result': result_float,
                'timestamp': datetime.now().isoformat(),
                'evaluation_count': len(self.optimization_history) + 1
            })
            
            # 检查早停
            if self.early_stopping:
                if self.early_stopping(result_float, validated_params):
                    logger.info("早停触发，停止优化")
                    # 这里可以通过异常或其他方式通知优化器停止
            
            return result_float
            
        except Exception as e:
            logger.error(f"评估过程中发生错误: {e}")
            return float('inf')
    
    def _evaluate_params_safe(self, params: Dict[str, float]) -> float:
        """线程安全的参数评估"""
        try:
            # 标准化参数
            normalized_params = normalize_params(params)
            
            # 验证参数类型
            validated_params = validate_param_types(normalized_params, self.param_space)
            
            result = self.base_evaluator.evaluate_mesh(validated_params)
            return float(result) if result != float('inf') else float('inf')
            
        except Exception as e:
            logger.error(f"并行参数评估失败: {e}")
            return float('inf')
    
    def _generate_random_params(self, n_samples: int) -> List[Dict[str, float]]:
        """生成随机参数组合"""
        param_sets = []
        bounds = self.param_space.get_bounds()
        param_names = self.param_space.get_param_names()
        param_types = self.param_space.get_param_types()
        
        # 设置随机种子
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
            value = result.x[i]
            # 标准化值
            if hasattr(value, 'item'):
                value = value.item()
            best_params[name] = value
        
        # 标准化最佳参数
        best_params = normalize_params(best_params)
        
        return {
            'best_params': best_params,
            'best_value': float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            'optimizer_name': optimizer_name,
            'skopt_result': result,
            'convergence_info': {
                'n_calls': len(result.func_vals),
                'best_iteration': int(np.argmin(result.func_vals)),
                'improvement_ratio': self._calculate_improvement_ratio(result.func_vals)
            }
        }
    
    def _calculate_improvement_ratio(self, func_vals: List[float]) -> float:
        """计算改进比例"""
        if len(func_vals) < 2:
            return 0.0
        
        initial_value = func_vals[0]
        final_value = min(func_vals)
        
        if initial_value == 0:
            return 0.0
        
        improvement = (initial_value - final_value) / initial_value
        return max(0.0, improvement)
    
    def _generate_optimization_report(self, result: Dict[str, Any]) -> str:
        """生成优化报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer_name = result['optimizer_name'].replace(' ', '_')
        report_dir = Path(f"optimization_reports/{timestamp}_{optimizer_name}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_file = report_dir / "optimization_report.txt"
        self._write_text_report(report_file, result)
        
        # 生成可视化图表
        if OPTIONAL_MODULES.get('skopt_available', False):
            try:
                self._generate_plots(result, report_dir)
            except Exception as e:
                logger.warning(f"生成图表失败: {e}")
        
        # 生成数据文件
        self._save_optimization_data(result, report_dir)
        
        logger.info(f"详细报告已保存到: {report_dir}")
        return str(report_dir)
    
    def _write_text_report(self, report_file: Path, result: Dict[str, Any]) -> None:
        """写入文本报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"优化报告 - {result['optimizer_name']}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"优化器: {result['optimizer']}\n")
            f.write(f"迭代次数: {result['n_calls']}\n")
            f.write(f"总评估次数: {result.get('total_evaluations', 'N/A')}\n")
            f.write(f"最佳目标值: {result['best_value']:.6f}\n\n")
            
            f.write("最佳参数:\n")
            for name, value in result['best_params'].items():
                f.write(f"  {name}: {value}\n")
            f.write("\n")
            
            # 收敛信息
            if 'convergence_info' in result:
                conv_info = result['convergence_info']
                f.write("收敛信息:\n")
                f.write(f"  最佳迭代: {conv_info.get('best_iteration', 'N/A')}\n")
                f.write(f"  改进比例: {conv_info.get('improvement_ratio', 0.0):.2%}\n")
                f.write("\n")
            
            # 缓存统计
            if self.cache:
                stats = self.cache.get_stats()
                f.write("缓存统计:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 早停信息
            if self.early_stopping and hasattr(self.early_stopping, 'should_stop'):
                if self.early_stopping.should_stop:
                    early_stop_info = self.early_stopping.get_best_result()
                    f.write("早停信息:\n")
                    for key, value in early_stop_info.items():
                        f.write(f"  {key}: {value}\n")
    
    def _generate_plots(self, result: Dict[str, Any], report_dir: Path) -> None:
        """生成可视化图表"""
        if not OPTIONAL_MODULES.get('skopt_available', False):
            return
        
        plt = OPTIONAL_MODULES['plt']
        
        try:
            # 收敛图
            if 'skopt_result' in result:
                plt.figure(figsize=(10, 6))
                OPTIONAL_MODULES['plot_convergence'](result['skopt_result'])
                plt.title(f"Convergence - {result['optimizer_name']}")
                plt.savefig(report_dir / "convergence.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 参数重要性图（如果数据足够）
                if result['n_calls'] >= 20:
                    try:
                        plt.figure(figsize=(12, 8))
                        OPTIONAL_MODULES['plot_objective'](result['skopt_result'])
                        plt.savefig(report_dir / "parameter_importance.png", dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        logger.warning(f"无法生成参数重要性图: {e}")
            
            # 优化历史图
            if self.optimization_history:
                self._plot_optimization_history(report_dir)
            
            # 早停历史图
            if self.early_stopping and hasattr(self.early_stopping, 'plot_history'):
                try:
                    self.early_stopping.plot_history(str(report_dir / "early_stopping_history.png"))
                except Exception as e:
                    logger.warning(f"无法生成早停历史图: {e}")
            
        except Exception as e:
            logger.warning(f"生成图表失败: {e}")
    
    @with_chinese_font
    def _plot_optimization_history(self, report_dir: Path) -> None:
        """绘制优化历史"""
        if not OPTIONAL_MODULES.get('skopt_available', False):
            return
        
        plt = OPTIONAL_MODULES['plt']
        
        try:
            results = [entry['result'] for entry in self.optimization_history]
            iterations = list(range(1, len(results) + 1))
            
            plt.figure(figsize=(12, 6))
            
            # 子图1: 目标值变化
            plt.subplot(1, 2, 1)
            plt.plot(iterations, results, 'b-', alpha=0.7, label='Objective Value')
            
            # 计算最佳值序列
            best_so_far = []
            current_best = float('inf')
            for result in results:
                if result < current_best:
                    current_best = result
                best_so_far.append(current_best)
            
            plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 改进分布
            plt.subplot(1, 2, 2)
            improvements = []
            for i in range(1, len(best_so_far)):
                if best_so_far[i] < best_so_far[i-1]:
                    improvements.append(i)
            
            if improvements:
                plt.scatter(improvements, [best_so_far[i] for i in improvements], 
                           c='red', s=50, alpha=0.7, label='Improvements')
                plt.plot(iterations, best_so_far, 'b-', alpha=0.5, label='Best Value')
                plt.xlabel('Iteration')
                plt.ylabel('Best Value')
                plt.title('Improvement Points')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(report_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"绘制优化历史失败: {e}")
    
    def _save_optimization_data(self, result: Dict[str, Any], report_dir: Path) -> None:
        """保存优化数据"""
        try:
            import json
            
            # 保存参数历史
            history_file = report_dir / "optimization_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
            
            # 保存最佳参数
            best_params_file = report_dir / "best_parameters.json"
            with open(best_params_file, 'w', encoding='utf-8') as f:
                json.dump(result['best_params'], f, indent=2, ensure_ascii=False)
            
            # 保存配置信息
            config_file = report_dir / "optimization_config.json"
            config_data = {
                'optimizer': result['optimizer'],
                'n_calls': result['n_calls'],
                'config': {
                    'early_stopping': self.config.early_stopping,
                    'use_cache': self.config.use_cache,
                    'random_state': self.config.random_state,
                    'patience': self.config.patience,
                    'min_delta': self.config.min_delta
                },
                'parameter_space': {
                    'param_names': self.param_space.get_param_names(),
                    'bounds': self.param_space.get_bounds(),
                    'param_types': [t.__name__ for t in self.param_space.get_param_types()]
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"保存优化数据失败: {e}")
    
    def sensitivity_analysis(self, 
                           best_params: Optional[Dict[str, float]] = None,
                           n_trials: int = 5,
                           noise_level: float = 0.1) -> Dict[str, List[tuple]]:
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
        
        # 确保 best_params 不为 None
        if best_params is None:
            raise ValueError("最佳参数为空，无法进行敏感性分析")
        
        logger.info("开始参数敏感性分析...")
        
        with performance_monitor("敏感性分析"):
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
                    range_size = int(max(1, param_value * noise_level))
                    min_val = max(low, int(param_value - range_size))
                    max_val = min(high, int(param_value + range_size))
                    test_values = np.linspace(min_val, max_val, n_trials, dtype=int)
                
                # 测试不同参数值的影响
                results = []
                for test_val in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = test_val
                    
                    try:
                        result = self.evaluator.evaluate_mesh(test_params)
                        results.append((test_val, result))
                        logger.debug(f"  {param_name}={test_val:.4f} -> {result:.4f}")
                    except Exception as e:
                        logger.warning(f"敏感性分析评估失败: {e}")
                        results.append((test_val, float('inf')))
                
                sensitivity_results[param_name] = results
            
            # 生成敏感性分析图表
            try:
                self._plot_sensitivity_analysis(sensitivity_results, best_params)
            except Exception as e:
                logger.warning(f"生成敏感性分析图表失败: {e}")
        
        logger.info("参数敏感性分析完成")
        return sensitivity_results
    
    @with_chinese_font
    def _plot_sensitivity_analysis(self,
                                  sensitivity_results: Dict[str, List],
                                  best_params: Dict[str, float]) -> None:
        """绘制敏感性分析图表"""
        if not OPTIONAL_MODULES.get('skopt_available', False):
            logger.warning("matplotlib不可用，跳过敏感性分析图表生成")
            return
        
        plt = OPTIONAL_MODULES['plt']
        
        try:
            n_params = len(sensitivity_results)
            if n_params == 0:
                return
                
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # 正确处理axes的类型
            if n_params == 1:
                axes_list = [axes]
            elif n_rows == 1:
                axes_list = list(axes) if n_cols > 1 else [axes]
            elif n_cols == 1:
                axes_list = list(axes) if n_rows > 1 else [axes]
            else:
                axes_list = axes.flatten()
            
            for i, (param_name, results) in enumerate(sensitivity_results.items()):
                ax = axes_list[i]
                
                test_values, objectives = zip(*results)
                ax.plot(test_values, objectives, 'o-', linewidth=2, markersize=6)
                ax.axvline(x=best_params[param_name], color='r', linestyle='--', linewidth=2, label='Best Value')
                ax.set_title(f'敏感性: {param_name}')
                ax.set_xlabel(param_name)
                ax.set_ylabel('目标值')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # 隐藏多余的子图
            for i in range(n_params, len(axes_list)):
                axes_list[i].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sensitivity_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            # 使用安全的显示和关闭函数
            if 'safe_show' in OPTIONAL_MODULES:
                OPTIONAL_MODULES['safe_show']()
            if 'safe_close' in OPTIONAL_MODULES:
                OPTIONAL_MODULES['safe_close']()
            
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
                'use_cache': self.config.use_cache,
                'available_modules': OPTIONAL_MODULES['available']
            }
        }
        
        if self.cache:
            summary['cache_stats'] = self.cache.get_stats()
        
        if self.early_stopping and hasattr(self.early_stopping, 'get_best_result'):
            summary['early_stopping_info'] = self.early_stopping.get_best_result()
        
        # 添加性能统计
        if self.optimization_history:
            results = [entry['result'] for entry in self.optimization_history]
            summary['performance_stats'] = {
                'best_value': min(results),
                'worst_value': max(results),
                'mean_value': np.mean(results),
                'std_value': np.std(results),
                'improvement_count': sum(1 for i in range(1, len(results)) if results[i] < results[i-1])
            }
        
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
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# 最佳网格参数 - {self.best_result['optimizer_name']}\n")
                f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
                f.write(f"# 最佳目标值: {self.best_result['best_value']:.6f}\n")
                f.write(f"# 总评估次数: {len(self.optimization_history)}\n\n")
                
                for key, value in self.best_result['best_params'].items():
                    f.write(f"{key} = {value}\n")
            
            logger.info(f"最佳参数已保存到: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"保存最佳参数失败: {e}")
            raise

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
        try:
            config_manager.load_config(config_file)
        except Exception as e:
            logger.warning(f"配置文件加载失败，使用默认配置: {e}")
    
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
    try:
        mesh_optimizer.save_best_params()
    except Exception as e:
        logger.warning(f"保存最佳参数失败: {e}")
    
    return result

def get_available_optimizers() -> List[str]:
    """获取可用的优化器列表"""
    available = ['genetic', 'parallel']  # 这些总是可用的
    
    if OPTIONAL_MODULES.get('skopt_available', False):
        available.extend(['bayesian', 'random', 'forest'])
    
    return sorted(available)

def check_dependencies() -> Dict[str, Any]:
    """检查依赖库状态"""
    return {
        'available_modules': OPTIONAL_MODULES['available'],
        'available_optimizers': get_available_optimizers(),
        'skopt_available': OPTIONAL_MODULES.get('skopt_available', False),
        'pandas_available': OPTIONAL_MODULES.get('pandas_available', False),
        'matplotlib_available': 'plt' in OPTIONAL_MODULES
    }

if __name__ == "__main__":
    # 示例用法
    logger.info("Ansa网格优化器示例")
    
    # 检查依赖
    deps = check_dependencies()
    print(f"可用优化器: {deps['available_optimizers']}")
    print(f"可用模块: {deps['available_modules']}")
    
    # 单个优化器测试
    try:
        result = optimize_mesh_parameters(
            n_calls=10,  # 减少迭代次数以便快速测试
            optimizer='genetic',  # 使用总是可用的遗传算法
            evaluator_type='mock'
        )
        
        print(f"\n最佳参数: {result['best_params']}")
        print(f"最佳值: {result['best_value']:.6f}")
        
    except Exception as e:
        logger.error(f"优化测试失败: {e}")
    
    print("\n示例运行完成!")