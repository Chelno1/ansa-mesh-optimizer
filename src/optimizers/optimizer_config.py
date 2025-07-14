"""
优化器配置类

定义优化器的配置参数和设置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """优化器配置类"""
    
    # 基础配置
    random_state: int = 42
    verbose: bool = True
    n_initial_points: int = 10
    
    # 遗传算法配置
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: str = 'tournament'
    tournament_size: int = 3
    elitism_ratio: float = 0.1
    
    # 并行配置
    n_workers: int = 4
    use_multiprocessing: bool = True
    
    # 收敛配置
    convergence_threshold: float = 1e-6
    max_stagnation_iterations: int = 20
    early_stopping: bool = True
    
    # 输出配置
    save_history: bool = True
    plot_convergence: bool = True
    save_plots: bool = True
    output_dir: str = "optimization_results"
    
    # 高级配置
    acquisition_function: str = 'EI'  # Expected Improvement
    n_restarts_optimizer: int = 5
    alpha: float = 1e-10
    
    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证配置"""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.random_state < 0:
            raise ValueError("random_state必须为非负整数")
        
        if self.n_initial_points < 1:
            raise ValueError("n_initial_points必须大于0")
        
        if self.population_size < 2:
            raise ValueError("population_size必须大于1")
        
        if self.n_generations < 1:
            raise ValueError("n_generations必须大于0")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate必须在[0,1]范围内")
        
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate必须在[0,1]范围内")
        
        if not 0 <= self.elitism_ratio <= 1:
            raise ValueError("elitism_ratio必须在[0,1]范围内")
        
        if self.n_workers < 1:
            raise ValueError("n_workers必须大于0")
        
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold必须大于0")
        
        if self.max_stagnation_iterations < 1:
            raise ValueError("max_stagnation_iterations必须大于0")
        
        if self.tournament_size < 2:
            raise ValueError("tournament_size必须大于1")
        
        valid_selection_methods = ['tournament', 'roulette', 'rank', 'random']
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method必须是{valid_selection_methods}之一")
        
        valid_acquisition_functions = ['EI', 'PI', 'UCB', 'LCB']
        if self.acquisition_function not in valid_acquisition_functions:
            raise ValueError(f"acquisition_function必须是{valid_acquisition_functions}之一")
    
    def _setup_logging(self):
        """设置日志级别"""
        if self.verbose:
            logging.getLogger('src.optimizers').setLevel(logging.INFO)
        else:
            logging.getLogger('src.optimizers').setLevel(logging.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_initial_points': self.n_initial_points,
            'population_size': self.population_size,
            'n_generations': self.n_generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_method': self.selection_method,
            'tournament_size': self.tournament_size,
            'elitism_ratio': self.elitism_ratio,
            'n_workers': self.n_workers,
            'use_multiprocessing': self.use_multiprocessing,
            'convergence_threshold': self.convergence_threshold,
            'max_stagnation_iterations': self.max_stagnation_iterations,
            'early_stopping': self.early_stopping,
            'save_history': self.save_history,
            'plot_convergence': self.plot_convergence,
            'save_plots': self.save_plots,
            'output_dir': self.output_dir,
            'acquisition_function': self.acquisition_function,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'alpha': self.alpha,
            'custom_params': self.custom_params.copy()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizerConfig':
        """从字典创建配置"""
        # 过滤掉不存在的字段
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"未知配置参数: {key}")
        
        # 重新验证配置
        self._validate_config()
    
    def get_optimizer_specific_config(self, optimizer_type: str) -> Dict[str, Any]:
        """获取特定优化器的配置"""
        base_config = {
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        if optimizer_type.lower() in ['bayesian', 'forest']:
            base_config.update({
                'n_initial_points': self.n_initial_points,
                'acquisition_function': self.acquisition_function,
                'n_restarts_optimizer': self.n_restarts_optimizer,
                'alpha': self.alpha
            })
        
        elif optimizer_type.lower() in ['genetic', 'ga']:
            base_config.update({
                'population_size': self.population_size,
                'n_generations': self.n_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'selection_method': self.selection_method,
                'tournament_size': self.tournament_size,
                'elitism_ratio': self.elitism_ratio,
                'convergence_threshold': self.convergence_threshold,
                'max_stagnation_iterations': self.max_stagnation_iterations,
                'early_stopping': self.early_stopping
            })
        
        elif optimizer_type.lower() == 'parallel':
            base_config.update({
                'n_workers': self.n_workers,
                'use_multiprocessing': self.use_multiprocessing
            })
        
        # 添加自定义参数
        base_config.update(self.custom_params)
        
        return base_config
    
    def copy(self) -> 'OptimizerConfig':
        """创建配置副本"""
        return OptimizerConfig.from_dict(self.to_dict())

@dataclass
class OptimizationResult:
    """
    统一的优化结果类
    
    结合了scikit-optimize兼容性、字典式访问和遗传算法特定功能
    """
    
    # 核心结果数据
    best_params: Dict[str, Any]
    best_value: float
    optimizer_name: str
    optimization_history: List[Dict[str, Any]]
    
    # 可选的元数据
    convergence_info: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    n_evaluations: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # 遗传算法特定属性
    generation_stats: Optional[List[Dict]] = None
    parameter_names: Optional[List[str]] = None
    parameter_ranges: Optional[List[tuple]] = None
    
    # 内部字典数据（用于向后兼容）
    _dict_data: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.n_evaluations is None:
            self.n_evaluations = len(self.optimization_history)
        
        # 设置scikit-optimize兼容属性
        self._setup_skopt_compatibility()
        
        # 设置字典兼容性数据
        self._setup_dict_compatibility()
    
    def _setup_skopt_compatibility(self):
        """设置scikit-optimize兼容属性"""
        # 从best_params提取参数值列表（如果有parameter_names）
        if self.parameter_names and isinstance(self.best_params, dict):
            self.x = [self.best_params.get(name, 0.0) for name in self.parameter_names]
        elif isinstance(self.best_params, list):
            self.x = self.best_params.copy()
        else:
            self.x = list(self.best_params.values()) if isinstance(self.best_params, dict) else []
        
        # 最佳分数（负值因为我们最小化）
        self.fun = self.best_value
        
        # 提取历史参数和分数
        self.x_iters = []
        self.func_vals = []
        
        for entry in self.optimization_history:
            # 提取参数
            if isinstance(entry.get('parameters'), dict):
                if self.parameter_names:
                    params = [entry['parameters'].get(name, 0.0) for name in self.parameter_names]
                else:
                    params = list(entry['parameters'].values())
            elif isinstance(entry.get('parameters'), list):
                params = entry['parameters']
            else:
                params = []
            
            self.x_iters.append(params)
            
            # 提取分数
            score = entry.get('result', entry.get('fitness', float('inf')))
            self.func_vals.append(score)
        
        # 其他scikit-optimize属性
        self.n_calls = self.n_evaluations or len(self.optimization_history)
        self.message = "Optimization completed successfully" if self.success else (self.error_message or "Optimization failed")
        
        # 兼容属性
        self.best_score = self.best_value
        self.history = self.optimization_history  # 保留原有字段以防兼容性问题
    
    def _setup_dict_compatibility(self):
        """设置字典兼容性数据"""
        self._dict_data = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'optimizer_name': self.optimizer_name,
            'optimization_history': self.optimization_history,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'n_evaluations': self.n_evaluations,
            'success': self.success,
            'error_message': self.error_message,
            'history': self.optimization_history,  # 保留原有字段以防兼容性问题
        }
        
        # 添加遗传算法特定数据
        if self.generation_stats is not None:
            self._dict_data.update({
                'generation_stats': self.generation_stats,
                'total_generations': len(self.generation_stats),
                'total_evaluations': self.n_calls,
            })
        
        if self.parameter_names is not None:
            self._dict_data['parameter_names'] = self.parameter_names
        
        if self.parameter_ranges is not None:
            self._dict_data['parameter_ranges'] = self.parameter_ranges
    
    # 字典式访问方法（向后兼容）
    def __getitem__(self, key):
        """支持字典式访问以保持向后兼容性"""
        if key in self._dict_data:
            return self._dict_data[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found")
    
    def get(self, key, default=None):
        """支持dict.get()式访问"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        """支持'in'操作符"""
        return key in self._dict_data or hasattr(self, key)
    
    def keys(self):
        """支持dict.keys()式访问"""
        return self._dict_data.keys()
    
    def values(self):
        """支持dict.values()式访问"""
        return self._dict_data.values()
    
    def items(self):
        """支持dict.items()式访问"""
        return self._dict_data.items()
    
    # 实用方法
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result_dict = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'optimizer_name': self.optimizer_name,
            'optimization_history': self.optimization_history,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'n_evaluations': self.n_evaluations,
            'success': self.success,
            'error_message': self.error_message
        }
        
        # 添加遗传算法特定数据
        if self.generation_stats is not None:
            result_dict['generation_stats'] = self.generation_stats
        if self.parameter_names is not None:
            result_dict['parameter_names'] = self.parameter_names
        if self.parameter_ranges is not None:
            result_dict['parameter_ranges'] = self.parameter_ranges
        
        return result_dict
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'OptimizationResult':
        """从字典创建结果"""
        # 提取核心参数
        core_params = {
            'best_params': result_dict.get('best_params', {}),
            'best_value': result_dict.get('best_value', float('inf')),
            'optimizer_name': result_dict.get('optimizer_name', 'Unknown'),
            'optimization_history': result_dict.get('optimization_history', []),
        }
        
        # 提取可选参数
        optional_params = {}
        for key in ['convergence_info', 'execution_time', 'n_evaluations', 'success', 'error_message',
                   'generation_stats', 'parameter_names', 'parameter_ranges']:
            if key in result_dict:
                optional_params[key] = result_dict[key]
        
        return cls(**core_params, **optional_params)
    
    @classmethod
    def from_genetic_result(cls, best_params: List[float], best_value: float,
                          optimization_history: List[Dict[str, Any]],
                          parameter_names: List[str], parameter_ranges: List[tuple],
                          generation_stats: Optional[List[Dict]] = None,
                          convergence_info: Optional[Dict] = None) -> 'OptimizationResult':
        """从遗传算法结果创建OptimizationResult（兼容genetic_optimizer.py的构造函数）"""
        
        # 将参数列表转换为字典
        best_params_dict = {parameter_names[i]: best_params[i] for i in range(len(parameter_names))}
        
        return cls(
            best_params=best_params_dict,
            best_value=best_value,
            optimizer_name='Genetic Algorithm',
            optimization_history=optimization_history,
            convergence_info=convergence_info,
            generation_stats=generation_stats,
            parameter_names=parameter_names,
            parameter_ranges=parameter_ranges,
            success=True
        )
    
    def get_improvement_ratio(self) -> float:
        """计算改进比例"""
        if len(self.optimization_history) < 2:
            return 0.0
        
        initial_value = self.optimization_history[0].get('result', float('inf'))
        final_value = self.best_value
        
        if initial_value == 0:
            return 0.0
        
        improvement = (initial_value - final_value) / initial_value
        return max(0.0, improvement)
    
    def get_convergence_rate(self) -> float:
        """计算收敛速度"""
        if not self.convergence_info or 'best_iteration' not in self.convergence_info:
            return 0.0
        
        best_iteration = self.convergence_info['best_iteration']
        total_iterations = self.n_evaluations or len(self.optimization_history)
        
        if total_iterations == 0:
            return 0.0
        
        return 1.0 - (best_iteration / total_iterations)
    
    def update_dict_data(self):
        """更新内部字典数据（在修改属性后调用）"""
        self._setup_dict_compatibility()

def create_default_config() -> OptimizerConfig:
    """创建默认配置"""
    return OptimizerConfig()

def create_fast_config() -> OptimizerConfig:
    """创建快速优化配置"""
    return OptimizerConfig(
        n_initial_points=5,
        population_size=20,
        n_generations=50,
        n_workers=2,
        early_stopping=True,
        max_stagnation_iterations=10
    )

def create_thorough_config() -> OptimizerConfig:
    """创建彻底优化配置"""
    return OptimizerConfig(
        n_initial_points=20,
        population_size=100,
        n_generations=200,
        n_workers=8,
        early_stopping=False,
        max_stagnation_iterations=50,
        convergence_threshold=1e-8
    )

def create_parallel_config(n_workers: Optional[int] = None) -> OptimizerConfig:
    """创建并行优化配置"""
    import multiprocessing as mp
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    return OptimizerConfig(
        n_workers=n_workers,
        use_multiprocessing=True,
        population_size=n_workers * 10,
        verbose=True
    )