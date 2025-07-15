"""
优化器策略实现

使用策略模式实现各种优化算法
"""

import numpy as np
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from .optimizer_config import OptimizationResult

logger = logging.getLogger(__name__)

# 安全导入可选依赖
def safe_import_skopt():
    """安全导入scikit-optimize"""
    try:
        from skopt import gp_minimize, forest_minimize, dummy_minimize
        from skopt.utils import use_named_args
        return {
            'available': True,
            'gp_minimize': gp_minimize,
            'forest_minimize': forest_minimize,
            'dummy_minimize': dummy_minimize,
            'use_named_args': use_named_args
        }
    except ImportError:
        return {
            'available': False,
            'gp_minimize': None,
            'forest_minimize': None,
            'dummy_minimize': None,
            'use_named_args': None
        }

SKOPT_MODULES = safe_import_skopt()

class OptimizerStrategy(ABC):
    """优化器策略抽象基类"""
    
    def __init__(self, param_space, evaluator, config):
        """
        初始化优化器策略
        
        Args:
            param_space: 参数空间
            evaluator: 评估器
            config: 配置对象
        """
        self.param_space = param_space
        self.evaluator = evaluator
        self.config = config
        self.optimization_history = []
    
    @abstractmethod
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """
        执行优化
        
        Args:
            n_calls: 迭代次数
            **kwargs: 其他参数
            
        Returns:
            优化结果对象
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取优化器名称"""
        pass
    
    def _evaluate_with_history(self, params: Dict[str, float]) -> float:
        """带历史记录的评估"""
        try:
            from datetime import datetime
            from utils.utils import normalize_params, validate_param_types
            
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
            
            return result_float
            
        except Exception as e:
            logger.error(f"评估过程中发生错误: {e}")
            return float('inf')
    
    def _format_result(self, best_params: Dict[str, Any], best_value: float,
                      additional_info: Optional[Dict[str, Any]] = None) -> 'OptimizationResult':
        """格式化优化结果"""
        from utils.utils import normalize_params
        from .optimizer_config import OptimizationResult
        
        # 标准化最佳参数
        normalized_best_params = normalize_params(best_params)
        
        # 准备基础参数
        base_params = {
            'best_params': normalized_best_params,
            'best_value': float(best_value),
            'optimizer_name': self.get_name(),
            'optimization_history': self.optimization_history.copy()
        }
        
        # 处理额外信息 - 只添加OptimizationResult支持的字段
        if additional_info:
            supported_fields = {
                'convergence_info', 'execution_time', 'n_evaluations',
                'success', 'error_message', 'generation_stats',
                'parameter_names', 'parameter_ranges',
                'skopt_result'
            }
            for key, value in additional_info.items():
                if key in supported_fields:
                    base_params[key] = value
        
        # 创建OptimizationResult对象
        result = OptimizationResult(**base_params)
        
        # 将不支持的额外信息添加到内部字典数据中
        if additional_info:
            for key, value in additional_info.items():
                if key not in base_params and not hasattr(result, key):
                    result._dict_data[key] = value
        
        # 更新字典数据
        result.update_dict_data()
        
        return result

class BayesianOptimizerStrategy(OptimizerStrategy):
    """贝叶斯优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行贝叶斯优化"""
        if not SKOPT_MODULES['available']:
            raise RuntimeError("贝叶斯优化需要安装scikit-optimize")
        
        # 安全获取函数
        use_named_args = SKOPT_MODULES.get('use_named_args')
        gp_minimize = SKOPT_MODULES.get('gp_minimize')
        
        if not use_named_args or not gp_minimize:
            raise RuntimeError("scikit-optimize模块不完整")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_history(params)
        
        result = gp_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result)
    
    def get_name(self) -> str:
        return "Bayesian Optimization"
    
    def _format_skopt_result(self, result) -> 'OptimizationResult':
        """格式化scikit-optimize结果"""
        param_names = self.param_space.get_param_names()
        
        best_params = {}
        for i, name in enumerate(param_names):
            value = result.x[i]
            if hasattr(value, 'item'):
                value = value.item()
            best_params[name] = value
        
        convergence_info = {
            'n_calls': len(result.func_vals),
            'best_iteration': int(np.argmin(result.func_vals)),
            'improvement_ratio': self._calculate_improvement_ratio(result.func_vals)
        }
        
        logger.info(f"格式化贝叶斯优化结果，skopt_result类型: {type(result)}")
        logger.info(f"skopt_result属性: func_vals={hasattr(result, 'func_vals')}, x_iters={hasattr(result, 'x_iters')}, space={hasattr(result, 'space')}")
        if hasattr(result, 'space') and result.space:
            logger.info(f"space维度: {result.space.n_dims}")
        
        optimization_result = self._format_result(
            best_params,
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )
        
        logger.info(f"创建的OptimizationResult有skopt_result: {hasattr(optimization_result, 'skopt_result')}")
        if hasattr(optimization_result, 'skopt_result'):
            logger.info(f"skopt_result是否为None: {optimization_result.skopt_result is None}")
        
        return optimization_result
    
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

class RandomOptimizerStrategy(OptimizerStrategy):
    """随机搜索优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行随机搜索优化"""
        if not SKOPT_MODULES['available']:
            raise RuntimeError("随机搜索需要安装scikit-optimize")
        
        # 安全获取函数
        use_named_args = SKOPT_MODULES.get('use_named_args')
        dummy_minimize = SKOPT_MODULES.get('dummy_minimize')
        
        if not use_named_args or not dummy_minimize:
            raise RuntimeError("scikit-optimize模块不完整")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_history(params)
        
        result = dummy_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result)
    
    def get_name(self) -> str:
        return "Random Search"
    
    def _format_skopt_result(self, result) -> 'OptimizationResult':
        """格式化scikit-optimize结果"""
        param_names = self.param_space.get_param_names()
        
        best_params = {}
        for i, name in enumerate(param_names):
            value = result.x[i]
            if hasattr(value, 'item'):
                value = value.item()
            best_params[name] = value
        
        convergence_info = {
            'n_calls': len(result.func_vals),
            'best_iteration': int(np.argmin(result.func_vals)),
            'improvement_ratio': self._calculate_improvement_ratio(result.func_vals)
        }

        optimization_result = self._format_result(
            best_params,
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )

        return optimization_result

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

class ForestOptimizerStrategy(OptimizerStrategy):
    """森林优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行森林优化"""
        if not SKOPT_MODULES['available']:
            raise RuntimeError("森林优化需要安装scikit-optimize")
        
        # 安全获取函数
        use_named_args = SKOPT_MODULES.get('use_named_args')
        forest_minimize = SKOPT_MODULES.get('forest_minimize')
        
        if not use_named_args or not forest_minimize:
            raise RuntimeError("scikit-optimize模块不完整")
        
        @use_named_args(self.param_space.to_skopt_space())
        def objective(**params):
            return self._evaluate_with_history(params)
        
        result = forest_minimize(
            objective,
            self.param_space.to_skopt_space(),
            n_calls=n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            **kwargs
        )
        
        return self._format_skopt_result(result)
    
    def get_name(self) -> str:
        return "Forest Optimization"
    
    def _format_skopt_result(self, result) -> 'OptimizationResult':
        """格式化scikit-optimize结果"""
        param_names = self.param_space.get_param_names()
        
        best_params = {}
        for i, name in enumerate(param_names):
            value = result.x[i]
            if hasattr(value, 'item'):
                value = value.item()
            best_params[name] = value
        
        convergence_info = {
            'n_calls': len(result.func_vals),
            'best_iteration': int(np.argmin(result.func_vals)),
            'improvement_ratio': self._calculate_improvement_ratio(result.func_vals)
        }
        
        optimization_result = self._format_result(
            best_params,
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )

        return optimization_result
    
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

class GeneticOptimizerStrategy(OptimizerStrategy):
    """遗传算法优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行遗传算法优化"""
        try:
            from optimizers.genetic_optimizer import GeneticOptimizer
            
            genetic_optimizer = GeneticOptimizer(
                param_space=self.param_space,
                evaluator=self.evaluator,
                config=self.config
            )
            
            result = genetic_optimizer.optimize(n_calls, **kwargs)
            
            # 更新历史记录
            if hasattr(genetic_optimizer, 'optimization_history'):
                self.optimization_history = getattr(genetic_optimizer, 'optimization_history', [])
            
            return result
            
        except ImportError as e:
            logger.error(f"遗传算法模块导入失败: {e}")
            raise RuntimeError("遗传算法优化器不可用")
    
    def get_name(self) -> str:
        return "Genetic Algorithm"

class ParallelGeneticOptimizerStrategy(OptimizerStrategy):
    """并行遗传算法优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行并行遗传算法优化"""
        from .parallel_genetic_optimizer import ParallelGeneticOptimizer, ParallelGeneticConfig
        
        # 创建并行遗传算法配置
        parallel_config = ParallelGeneticConfig(
            # 基础遗传算法参数
            population_size=getattr(self.config, 'population_size', 50),
            max_generations=getattr(self.config, 'n_generations', 100),
            crossover_rate=getattr(self.config, 'crossover_rate', 0.8),
            mutation_rate=getattr(self.config, 'mutation_rate', 0.1),
            elite_size=max(1, int(getattr(self.config, 'population_size', 50) * getattr(self.config, 'elitism_ratio', 0.1))),
            tournament_size=getattr(self.config, 'tournament_size', 3),
            
            # 并行配置
            n_workers=getattr(self.config, 'n_workers', 4),
            use_multiprocessing=getattr(self.config, 'use_multiprocessing', True),
            parallel_evaluation=True,
            parallel_diversity=True,
            parallel_evolution=True,
            
            # 批处理配置
            evaluation_batch_size=max(1, getattr(self.config, 'n_workers', 4) * 2),
            evolution_batch_size=max(1, getattr(self.config, 'n_workers', 4) * 4),
            auto_batch_size=True,
            
            # 性能优化
            vectorized_operations=True,
            cache_evaluations=True,
            lazy_evaluation=True,
            
            # 内存管理
            max_memory_usage=0.8,
            memory_monitoring=True,
            
            # 容错配置
            fault_tolerance=True,
            max_retries=3,
            timeout_seconds=300,
            
            # 性能监控
            performance_monitoring=True,
            log_performance=getattr(self.config, 'verbose', True),
            
            # 收敛配置
            convergence_threshold=getattr(self.config, 'convergence_threshold', 1e-6),
            max_stagnation_iterations=getattr(self.config, 'max_stagnation_iterations', 20),
            early_stopping=getattr(self.config, 'early_stopping', True),
            
            # 多样性保持
            diversity_preservation=True,
            
            # 随机种子
            random_state=getattr(self.config, 'random_state', 42)
        )
        
        # 创建并行遗传算法优化器
        parallel_optimizer = ParallelGeneticOptimizer(
            param_space=self.param_space,
            evaluator=self.evaluator,
            parallel_config=parallel_config
        )
        
        logger.info(f"开始并行遗传算法优化: 工作进程数={parallel_config.n_workers}, 种群大小={parallel_config.population_size}")
        
        # 执行优化
        result = parallel_optimizer.optimize(n_calls, **kwargs)
        
        # 更新历史记录
        if hasattr(parallel_optimizer, 'optimization_history'):
            self.optimization_history = getattr(parallel_optimizer, 'optimization_history', [])
        
        # 添加并行特定信息
        if hasattr(result, '_dict_data'):
            result._dict_data.update({
                'parallel_config': {
                    'n_workers': parallel_config.n_workers,
                    'parallel_evaluation': parallel_config.parallel_evaluation,
                    'parallel_diversity': parallel_config.parallel_diversity,
                    'vectorized_operations': parallel_config.vectorized_operations,
                    'cache_evaluations': parallel_config.cache_evaluations
                }
            })
        
        # 获取性能摘要
        if hasattr(parallel_optimizer, 'performance_monitor'):
            performance_summary = parallel_optimizer.performance_monitor.get_performance_summary()
            logger.info(f"并行遗传算法性能摘要: {performance_summary}")
            
            if hasattr(result, '_dict_data'):
                result._dict_data['performance_summary'] = performance_summary
        
        return result
    
    def get_name(self) -> str:
        return "Parallel Genetic Algorithm"


class ParallelOptimizerStrategy(OptimizerStrategy):
    """并行随机搜索策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> 'OptimizationResult':
        """执行并行随机搜索"""
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
        
        # 确保best_params不为None
        if best_params is None:
            best_params = {}
            best_value = float('inf')
        
        return self._format_result(
            best_params,
            best_value,
            {'all_results': all_results}
        )
    
    def get_name(self) -> str:
        return "Parallel Random Search"
    
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
            for i, (name, bound, param_type) in enumerate(zip(param_names, bounds, param_types)):
                if isinstance(bound, tuple) and len(bound) == 2:
                    low, high = bound
                    if param_type == int or param_type == 'integer':
                        params[name] = np.random.randint(int(low), int(high) + 1)
                    else:
                        params[name] = np.random.uniform(float(low), float(high))
            param_sets.append(params)
        
        return param_sets
    
    def _evaluate_params_safe(self, params: Dict[str, float]) -> float:
        """线程安全的参数评估"""
        try:
            from utils.utils import normalize_params, validate_param_types
            
            # 标准化参数
            normalized_params = normalize_params(params)
            
            # 验证参数类型
            validated_params = validate_param_types(normalized_params, self.param_space)
            
            # 使用基础评估器（避免缓存问题）
            if hasattr(self.evaluator, 'base_evaluator'):
                result = self.evaluator.base_evaluator.evaluate_mesh(validated_params)
            else:
                result = self.evaluator.evaluate_mesh(validated_params)
            
            return float(result) if result != float('inf') else float('inf')
            
        except Exception as e:
            logger.error(f"并行参数评估失败: {e}")
            return float('inf')

class OptimizerFactory:
    """优化器工厂类"""
    
    _strategies = {
        'bayesian': BayesianOptimizerStrategy,
        'random': RandomOptimizerStrategy,
        'forest': ForestOptimizerStrategy,
        'genetic': GeneticOptimizerStrategy,
        'ga': GeneticOptimizerStrategy,  # 别名
        'parallel_genetic': ParallelGeneticOptimizerStrategy,
        'parallel_ga': ParallelGeneticOptimizerStrategy,  # 别名
        'parallel': ParallelOptimizerStrategy
    }
    
    @classmethod
    def create_optimizer(cls, optimizer_type: str, param_space, evaluator, config) -> OptimizerStrategy:
        """
        创建优化器策略实例
        
        Args:
            optimizer_type: 优化器类型
            param_space: 参数空间
            evaluator: 评估器
            config: 配置对象
            
        Returns:
            优化器策略实例
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"不支持的优化器类型: {optimizer_type}. 可用类型: {available}")
        
        # 检查依赖
        if optimizer_type in ['bayesian', 'random', 'forest'] and not SKOPT_MODULES['available']:
            raise RuntimeError(f"优化器 {optimizer_type} 需要安装 scikit-optimize")
        
        strategy_class = cls._strategies[optimizer_type]
        return strategy_class(param_space, evaluator, config)
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """获取可用的优化器列表"""
        available = ['genetic', 'parallel_genetic', 'parallel']  # 这些总是可用的
        
        if SKOPT_MODULES['available']:
            available.extend(['bayesian', 'random', 'forest'])
        
        return sorted(available)
    
    @classmethod
    def register_optimizer(cls, name: str, strategy_class):
        """注册新的优化器策略"""
        if not issubclass(strategy_class, OptimizerStrategy):
            raise ValueError("策略类必须继承自 OptimizerStrategy")
        
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"已注册优化器策略: {name}")