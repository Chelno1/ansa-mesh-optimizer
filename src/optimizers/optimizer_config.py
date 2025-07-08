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
    """优化结果类"""
    
    best_params: Dict[str, Any]
    best_value: float
    optimizer_name: str
    optimization_history: List[Dict[str, Any]]
    convergence_info: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    n_evaluations: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.n_evaluations is None:
            self.n_evaluations = len(self.optimization_history)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
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
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'OptimizationResult':
        """从字典创建结果"""
        return cls(**result_dict)
    
    def get_improvement_ratio(self) -> float:
        """计算改进比例"""
        if len(self.optimization_history) < 2:
            return 0.0
        
        initial_value = self.optimization_history[0]['result']
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