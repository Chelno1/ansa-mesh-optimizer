#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制模块 - 增强版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 自适应策略，内存优化，多种停止条件
"""

import logging
import time
import numpy as np
from typing import List, Optional, Dict, Any, Callable, Tuple
from datetime import datetime
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

# 安全导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib不可用，无法生成早停图表")

class EarlyStopping:
    """早停机制类 - 增强版本"""
    
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.01, 
                 restore_best_weights: bool = True,
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 verbose: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍不改善的迭代次数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
            mode: 监控模式 ('min' 或 'max')
            baseline: 基线值，如果设置，只有超过基线才算改善
            verbose: 是否输出详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode.lower()
        self.baseline = baseline
        self.verbose = verbose
        
        # 验证模式
        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        
        # 初始化状态
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_iteration = 0
        self.best_params = None
        self.wait = 0
        self.stopped_iteration = 0
        self.should_stop = False
        
        # 历史记录（使用deque限制内存使用）
        self.history: deque = deque(maxlen=1000)  # 最多保存1000个历史值
        self.improvement_history: List[int] = []  # 改善的迭代次数
        
        # 统计信息
        self.total_iterations = 0
        self.improvement_count = 0
        self.start_time = time.time()
        
        if self.verbose:
            logger.info(f"早停机制初始化: patience={patience}, min_delta={min_delta}, mode={mode}")
    
    def __call__(self, current_value: float, current_params: Optional[Dict] = None) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_value: 当前目标值
            current_params: 当前参数（可选）
            
        Returns:
            是否应该停止训练
        """
        self.total_iterations += 1
        current_iteration = len(self.history)
        
        # 添加到历史
        self.history.append({
            'value': current_value,
            'iteration': current_iteration,
            'timestamp': time.time(),
            'params': current_params.copy() if current_params else None
        })
        
        # 检查是否有改善
        improved = self._check_improvement(current_value)
        
        if improved:
            self.best_value = current_value
            self.best_iteration = current_iteration
            if current_params is not None and self.restore_best_weights:
                self.best_params = current_params.copy()
            self.wait = 0
            self.improvement_count += 1
            self.improvement_history.append(current_iteration)
            
            if self.verbose:
                logger.debug(f"发现改善: {current_value:.6f} (迭代 {current_iteration})")
        else:
            self.wait += 1
            if self.verbose:
                logger.debug(f"无改善，等待计数: {self.wait}/{self.patience}")
        
        # 检查是否应该停止
        if self.wait >= self.patience:
            self.stopped_iteration = current_iteration
            self.should_stop = True
            
            if self.verbose:
                logger.info(f"早停触发：在第{self.stopped_iteration}次迭代，"
                          f"最佳值: {self.best_value:.6f} (第{self.best_iteration}次迭代)")
            
            return True
        
        return False
    
    def _check_improvement(self, current_value: float) -> bool:
        """检查当前值是否是改善"""
        # 检查基线
        if self.baseline is not None:
            if self.mode == 'min' and current_value >= self.baseline:
                return False
            if self.mode == 'max' and current_value <= self.baseline:
                return False
        
        # 检查相对于最佳值的改善
        if self.mode == 'min':
            return current_value < self.best_value - self.min_delta
        else:
            return current_value > self.best_value + self.min_delta
    
    def get_best_result(self) -> Dict[str, Any]:
        """
        获取最佳结果
        
        Returns:
            包含最佳结果信息的字典
        """
        return {
            'best_value': self.best_value,
            'best_iteration': self.best_iteration,
            'best_params': self.best_params,
            'stopped_iteration': self.stopped_iteration,
            'total_iterations': self.total_iterations,
            'improvement_count': self.improvement_count,
            'wait_count': self.wait,
            'should_stop': self.should_stop,
            'improvement_rate': self.improvement_count / self.total_iterations if self.total_iterations > 0 else 0,
            'execution_time': time.time() - self.start_time
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        if not self.history:
            return {'error': 'No history available'}
        
        values = [entry['value'] for entry in self.history]
        
        stats = {
            'total_iterations': len(self.history),
            'best_value': self.best_value,
            'worst_value': max(values) if self.mode == 'min' else min(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'improvement_count': self.improvement_count,
            'improvement_rate': self.improvement_count / len(self.history),
            'stagnation_periods': self._analyze_stagnation_periods(),
            'convergence_trend': self._analyze_convergence_trend()
        }
        
        return stats
    
    def _analyze_stagnation_periods(self) -> List[Dict[str, Any]]:
        """分析停滞期"""
        if len(self.improvement_history) < 2:
            return []
        
        stagnation_periods = []
        for i in range(1, len(self.improvement_history)):
            period_start = self.improvement_history[i-1]
            period_end = self.improvement_history[i]
            period_length = period_end - period_start
            
            if period_length > 1:  # 只记录长度大于1的停滞期
                stagnation_periods.append({
                    'start': period_start,
                    'end': period_end,
                    'length': period_length
                })
        
        return stagnation_periods
    
    def _analyze_convergence_trend(self) -> str:
        """分析收敛趋势"""
        if len(self.history) < 5:
            return 'insufficient_data'
        
        recent_values = [entry['value'] for entry in list(self.history)[-5:]]
        
        # 计算趋势
        if self.mode == 'min':
            if recent_values[-1] < recent_values[0]:
                return 'improving'
            elif recent_values[-1] > recent_values[0]:
                return 'deteriorating'
        else:
            if recent_values[-1] > recent_values[0]:
                return 'improving'
            elif recent_values[-1] < recent_values[0]:
                return 'deteriorating'
        
        # 检查稳定性
        variance = np.var(recent_values)
        if variance < self.min_delta ** 2:
            return 'stable'
        
        return 'fluctuating'
    
    def plot_history(self, save_path: Optional[str] = None, show_improvements: bool = True):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径（可选）
            show_improvements: 是否显示改善点
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，无法绘制历史图")
            return
        
        if not self.history:
            logger.warning("没有历史数据可绘制")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 提取数据
            iterations = [entry['iteration'] for entry in self.history]
            values = [entry['value'] for entry in self.history]
            
            # 主要优化曲线
            ax1.plot(iterations, values, 'b-', linewidth=1, alpha=0.7, label='目标值')
            
            # 最佳值线
            best_values = []
            current_best = float('inf') if self.mode == 'min' else float('-inf')
            
            for value in values:
                if self.mode == 'min':
                    if value < current_best:
                        current_best = value
                else:
                    if value > current_best:
                        current_best = value
                best_values.append(current_best)
            
            ax1.plot(iterations, best_values, 'r-', linewidth=2, label='历史最佳值')
            
            # 标记改善点
            if show_improvements and self.improvement_history:
                improvement_values = [values[i] for i in self.improvement_history if i < len(values)]
                ax1.scatter(self.improvement_history[:len(improvement_values)], 
                          improvement_values, 
                          c='green', s=50, alpha=0.7, label='改善点', zorder=5)
            
            # 标记早停点
            if self.should_stop:
                ax1.axvline(x=self.stopped_iteration, color='orange', linestyle='--',
                           label=f'早停点 (迭代 {self.stopped_iteration})')
            
            # 标记最佳点
            ax1.axvline(x=self.best_iteration, color='red', linestyle='--', alpha=0.7,
                       label=f'最佳值点 (迭代 {self.best_iteration})')
            
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('目标值')
            ax1.set_title('优化历史与早停分析')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 改善间隔分析
            if len(self.improvement_history) > 1:
                intervals = np.diff(self.improvement_history)
                ax2.bar(range(len(intervals)), intervals, alpha=0.7, color='skyblue')
                ax2.set_xlabel('改善序号')
                ax2.set_ylabel('改善间隔 (迭代次数)')
                ax2.set_title('改善间隔分析')
                ax2.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_interval = np.mean(intervals)
                ax2.axhline(y=mean_interval, color='red', linestyle='--', 
                           label=f'平均间隔: {mean_interval:.1f}')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, '改善次数不足，无法分析间隔', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('改善间隔分析 (数据不足)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"早停历史图已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"绘制历史图失败: {e}")
    
    def reset(self):
        """重置早停状态"""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_iteration = 0
        self.best_params = None
        self.wait = 0
        self.stopped_iteration = 0
        self.should_stop = False
        self.history.clear()
        self.improvement_history.clear()
        self.total_iterations = 0
        self.improvement_count = 0
        self.start_time = time.time()
        
        if self.verbose:
            logger.info("早停状态已重置")
    
    def save_state(self, filepath: str):
        """保存早停状态到文件"""
        try:
            import pickle
            
            state = {
                'config': {
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                    'mode': self.mode,
                    'baseline': self.baseline
                },
                'state': {
                    'best_value': self.best_value,
                    'best_iteration': self.best_iteration,
                    'best_params': self.best_params,
                    'wait': self.wait,
                    'should_stop': self.should_stop,
                    'total_iterations': self.total_iterations,
                    'improvement_count': self.improvement_count
                },
                'history': list(self.history),
                'improvement_history': self.improvement_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"早停状态已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存早停状态失败: {e}")
    
    def load_state(self, filepath: str):
        """从文件加载早停状态"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复配置
            config = state['config']
            self.patience = config['patience']
            self.min_delta = config['min_delta']
            self.mode = config['mode']
            self.baseline = config['baseline']
            
            # 恢复状态
            state_data = state['state']
            self.best_value = state_data['best_value']
            self.best_iteration = state_data['best_iteration']
            self.best_params = state_data['best_params']
            self.wait = state_data['wait']
            self.should_stop = state_data['should_stop']
            self.total_iterations = state_data['total_iterations']
            self.improvement_count = state_data['improvement_count']
            
            # 恢复历史
            self.history = deque(state['history'], maxlen=1000)
            self.improvement_history = state['improvement_history']
            
            logger.info(f"早停状态已加载: {filepath}")
            
        except Exception as e:
            logger.error(f"加载早停状态失败: {e}")

class AdaptiveEarlyStopping(EarlyStopping):
    """自适应早停机制 - 增强版本"""
    
    def __init__(self, 
                 initial_patience: int = 5, 
                 min_delta: float = 0.01,
                 patience_factor: float = 1.5, 
                 max_patience: int = 20,
                 adaptation_threshold: int = 3,
                 **kwargs):
        """
        初始化自适应早停机制
        
        Args:
            initial_patience: 初始耐心值
            min_delta: 最小改善阈值
            patience_factor: 耐心增长因子
            max_patience: 最大耐心值
            adaptation_threshold: 触发自适应的改善次数阈值
            **kwargs: 其他参数传递给父类
        """
        super().__init__(initial_patience, min_delta, **kwargs)
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.adaptation_threshold = adaptation_threshold
        
        # 自适应状态
        self.adaptation_count = 0
        self.patience_history = [initial_patience]
        
        if self.verbose:
            logger.info(f"自适应早停初始化: 初始耐心={initial_patience}, "
                      f"最大耐心={max_patience}, 适应阈值={adaptation_threshold}")
    
    def __call__(self, current_value: float, current_params: Optional[Dict] = None) -> bool:
        """自适应早停检查"""
        # 调用父类方法记录历史
        super().__call__(current_value, current_params)
        
        # 检查是否有改善（重新计算，因为父类已经更新了状态）
        if self.wait == 0:  # 刚刚发生了改善
            self.improvement_count += 1
            
            # 检查是否需要自适应调整
            if self.improvement_count % self.adaptation_threshold == 0:
                self._adapt_patience()
        
        # 返回是否应该停止（使用更新后的patience）
        if self.wait >= self.patience:
            self.stopped_iteration = len(self.history) - 1
            self.should_stop = True
            
            if self.verbose:
                logger.info(f"自适应早停触发：在第{self.stopped_iteration}次迭代，"
                          f"最终耐心值={self.patience}")
            return True
        
        return False
    
    def _adapt_patience(self):
        """自适应调整耐心值"""
        old_patience = self.patience
        
        # 计算新的耐心值
        new_patience = min(
            int(self.patience * self.patience_factor), 
            self.max_patience
        )
        
        if new_patience > old_patience:
            self.patience = new_patience
            self.adaptation_count += 1
            self.patience_history.append(new_patience)
            
            if self.verbose:
                logger.info(f"自适应调整: 耐心值从 {old_patience} 增加到 {new_patience} "
                          f"(第{self.adaptation_count}次调整)")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """获取自适应统计信息"""
        return {
            'adaptation_count': self.adaptation_count,
            'initial_patience': self.initial_patience,
            'current_patience': self.patience,
            'max_patience': self.max_patience,
            'patience_history': self.patience_history.copy(),
            'adaptation_efficiency': self.improvement_count / self.adaptation_count if self.adaptation_count > 0 else 0
        }

class ConvergenceDetector:
    """收敛检测器 - 增强版本"""
    
    def __init__(self, 
                 window_size: int = 10, 
                 tolerance: float = 1e-6,
                 trend_threshold: float = 1e-4,
                 min_samples: int = 5):
        """
        初始化收敛检测器
        
        Args:
            window_size: 滑动窗口大小
            tolerance: 收敛容忍度
            trend_threshold: 趋势变化阈值
            min_samples: 最小样本数
        """
        self.window_size = window_size
        self.tolerance = tolerance
        self.trend_threshold = trend_threshold
        self.min_samples = min_samples
        
        self.values: deque = deque(maxlen=window_size)
        self.convergence_history: List[bool] = []
        
    def add_value(self, value: float) -> Tuple[bool, Dict[str, Any]]:
        """
        添加新值并检查收敛
        
        Args:
            value: 新的目标值
            
        Returns:
            (是否已收敛, 收敛信息字典)
        """
        self.values.append(value)
        
        convergence_info = {
            'converged': False,
            'variance': None,
            'trend': None,
            'relative_change': None,
            'samples_count': len(self.values)
        }
        
        # 检查收敛
        if len(self.values) >= self.min_samples:
            recent_values = np.array(list(self.values))
            
            # 计算方差
            variance = np.var(recent_values)
            convergence_info['variance'] = variance
            
            # 计算相对变化
            if len(self.values) > 1:
                relative_change = abs(self.values[-1] - self.values[-2]) / (abs(self.values[-1]) + 1e-8)
                convergence_info['relative_change'] = relative_change
            else:
                relative_change = float('inf')
            
            # 计算趋势
            trend_slope = self._calculate_trend()
            convergence_info['trend'] = trend_slope
            
            # 判断收敛
            variance_converged = variance < self.tolerance
            change_converged = relative_change < self.tolerance
            trend_converged = abs(trend_slope) < self.trend_threshold
            
            converged = variance_converged and change_converged and trend_converged
            convergence_info['converged'] = converged
            
            self.convergence_history.append(converged)
            
            if converged:
                logger.debug(f"检测到收敛: 方差={variance:.2e}, "
                           f"相对变化={relative_change:.2e}, 趋势={trend_slope:.2e}")
        
        return convergence_info['converged'], convergence_info
    
    def _calculate_trend(self) -> float:
        """计算趋势斜率"""
        if len(self.values) < 2:
            return 0.0
        
        # 使用线性回归计算趋势
        x = np.arange(len(self.values))
        y = np.array(list(self.values))
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def get_trend_description(self) -> str:
        """获取趋势描述"""
        if len(self.values) < self.min_samples:
            return "insufficient_data"
        
        slope = self._calculate_trend()
        
        if abs(slope) < self.trend_threshold:
            return "stable"
        elif slope < -self.trend_threshold:
            return "improving"  # 假设目标是最小化
        else:
            return "deteriorating"
    
    def reset(self):
        """重置检测器"""
        self.values.clear()
        self.convergence_history.clear()

class MultiCriteriaEarlyStopping:
    """多标准早停机制"""
    
    def __init__(self, 
                 criteria: List[Dict[str, Any]],
                 combination_mode: str = 'any'):
        """
        初始化多标准早停
        
        Args:
            criteria: 早停标准列表，每个标准是一个配置字典
            combination_mode: 标准组合模式 ('any', 'all', 'majority')
        """
        self.combination_mode = combination_mode
        self.early_stoppers = []
        
        # 创建各个早停器
        for criterion in criteria:
            criterion_type = criterion.get('type', 'standard')
            
            if criterion_type == 'standard':
                stopper = EarlyStopping(**{k: v for k, v in criterion.items() if k != 'type'})
            elif criterion_type == 'adaptive':
                stopper = AdaptiveEarlyStopping(**{k: v for k, v in criterion.items() if k != 'type'})
            else:
                raise ValueError(f"Unknown early stopping type: {criterion_type}")
            
            self.early_stoppers.append(stopper)
        
        logger.info(f"多标准早停初始化: {len(self.early_stoppers)} 个标准, 组合模式={combination_mode}")
    
    def __call__(self, current_value: float, current_params: Optional[Dict] = None) -> bool:
        """检查是否应该早停"""
        stop_signals = []
        
        for stopper in self.early_stoppers:
            should_stop = stopper(current_value, current_params)
            stop_signals.append(should_stop)
        
        # 根据组合模式决定
        if self.combination_mode == 'any':
            return any(stop_signals)
        elif self.combination_mode == 'all':
            return all(stop_signals)
        elif self.combination_mode == 'majority':
            return sum(stop_signals) > len(stop_signals) / 2
        else:
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取所有早停器的状态"""
        status = {
            'combination_mode': self.combination_mode,
            'stoppers': []
        }
        
        for i, stopper in enumerate(self.early_stoppers):
            stopper_status = stopper.get_best_result()
            stopper_status['index'] = i
            stopper_status['type'] = type(stopper).__name__
            status['stoppers'].append(stopper_status)
        
        return status

def create_early_stopping(config) -> EarlyStopping:
    """
    根据配置创建早停机制
    
    Args:
        config: 配置对象
        
    Returns:
        早停机制实例
    """
    if hasattr(config, 'adaptive_early_stopping') and config.adaptive_early_stopping:
        return AdaptiveEarlyStopping(
            initial_patience=config.patience,
            min_delta=config.min_delta,
            verbose=config.verbose
        )
    else:
        return EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=config.verbose
        )

# 工厂函数
def create_early_stopping_with_criteria(criteria_list: List[Dict[str, Any]], 
                                       combination_mode: str = 'any') -> MultiCriteriaEarlyStopping:
    """
    创建多标准早停机制
    
    Args:
        criteria_list: 早停标准列表
        combination_mode: 组合模式
        
    Returns:
        多标准早停机制实例
    """
    return MultiCriteriaEarlyStopping(criteria_list, combination_mode)

if __name__ == "__main__":
    # 测试早停机制
    print("=== 早停机制测试 ===")
    
    # 测试标准早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=True)
    
    # 模拟优化过程
    test_values = [10.0, 8.5, 7.2, 6.8, 6.7, 6.65, 6.64, 6.63, 6.63, 6.63, 6.63]
    
    print("测试标准早停:")
    for i, value in enumerate(test_values):
        should_stop = early_stopping(value)
        print(f"迭代 {i}: 值={value:.3f}, 应停止={should_stop}")
        if should_stop:
            break
    
    print(f"\n最佳结果: {early_stopping.get_best_result()}")
    print(f"统计信息: {early_stopping.get_statistics()}")
    
    # 测试自适应早停
    print("\n" + "="*50)
    print("测试自适应早停:")
    
    adaptive_stopping = AdaptiveEarlyStopping(
        initial_patience=3, 
        adaptation_threshold=2,
        verbose=True
    )
    
    # 模拟有更多改善的优化过程
    test_values_2 = [15.0, 12.0, 10.0, 9.5, 9.4, 8.0, 7.5, 7.4, 7.35, 7.3, 7.25, 7.2, 6.0, 5.8, 5.75, 5.74, 5.73]
    
    for i, value in enumerate(test_values_2):
        should_stop = adaptive_stopping(value)
        print(f"迭代 {i}: 值={value:.3f}, 应停止={should_stop}, 当前耐心={adaptive_stopping.patience}")
        if should_stop:
            break
    
    print(f"\n自适应统计: {adaptive_stopping.get_adaptation_stats()}")
    
    # 测试收敛检测器
    print("\n" + "="*50)
    print("测试收敛检测器:")
    
    convergence_detector = ConvergenceDetector(window_size=5, tolerance=1e-3)
    
    convergence_values = [10.0, 5.0, 2.5, 1.25, 1.0, 1.001, 1.0005, 1.0002, 1.0001, 1.0]
    
    for i, value in enumerate(convergence_values):
        converged, info = convergence_detector.add_value(value)
        print(f"迭代 {i}: 值={value:.4f}, 收敛={converged}, 趋势={convergence_detector.get_trend_description()}")
        if converged:
            print("检测到收敛!")
            break
    
    # 绘制历史图（如果matplotlib可用）
    if MATPLOTLIB_AVAILABLE:
        early_stopping.plot_history("test_early_stopping.png")
    
    print("\n早停机制测试完成!")