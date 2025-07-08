"""
优化器策略实现

使用策略模式实现各种优化算法
"""

import numpy as np
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            n_calls: 迭代次数
            **kwargs: 其他参数
            
        Returns:
            优化结果字典
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
            from src.utils.utils import normalize_params, validate_param_types
            
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
                      additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """格式化优化结果"""
        from src.utils.utils import normalize_params
        
        # 标准化最佳参数
        normalized_best_params = normalize_params(best_params)
        
        result = {
            'best_params': normalized_best_params,
            'best_value': float(best_value),
            'optimizer_name': self.get_name(),
            'optimization_history': self.optimization_history.copy()
        }
        
        if additional_info:
            result.update(additional_info)
        
        return result

class BayesianOptimizerStrategy(OptimizerStrategy):
    """贝叶斯优化策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
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
    
    def _format_skopt_result(self, result) -> Dict[str, Any]:
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
        
        return self._format_result(
            best_params, 
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )
    
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
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
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
    
    def _format_skopt_result(self, result) -> Dict[str, Any]:
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
        
        return self._format_result(
            best_params, 
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )
    
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
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
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
    
    def _format_skopt_result(self, result) -> Dict[str, Any]:
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
        
        return self._format_result(
            best_params, 
            float(result.fun) if hasattr(result.fun, 'item') else result.fun,
            {
                'skopt_result': result,
                'convergence_info': convergence_info
            }
        )
    
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
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """执行遗传算法优化"""
        try:
            from src.core.genetic_optimizer_improved import GeneticOptimizer
            
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

class ParallelOptimizerStrategy(OptimizerStrategy):
    """并行随机搜索策略"""
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
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
            from src.utils.utils import normalize_params, validate_param_types
            
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
        available = ['genetic', 'parallel']  # 这些总是可用的
        
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