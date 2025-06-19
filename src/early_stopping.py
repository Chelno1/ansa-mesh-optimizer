#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制模块

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.01, restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍不改善的迭代次数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf')
        self.best_iteration = 0
        self.best_params = None
        self.wait = 0
        self.stopped_iteration = 0
        self.should_stop = False
        
        self.history: List[float] = []
    
    def __call__(self, current_value: float, current_params: Optional[dict] = None) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_value: 当前目标值
            current_params: 当前参数（可选）
            
        Returns:
            是否应该停止训练
        """
        self.history.append(current_value)
        
        # 检查是否有改善
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.best_iteration = len(self.history) - 1
            if current_params is not None:
                self.best_params = current_params.copy()
            self.wait = 0
            logger.debug(f"发现改善: {current_value:.6f}")
        else:
            self.wait += 1
            logger.debug(f"无改善，等待计数: {self.wait}/{self.patience}")
        
        # 检查是否应该停止
        if self.wait >= self.patience:
            self.stopped_iteration = len(self.history) - 1
            self.should_stop = True
            logger.info(f"早停触发：在第{self.stopped_iteration}次迭代，最佳值: {self.best_value:.6f}")
            return True
        
        return False
    
    def get_best_result(self) -> dict:
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
            'total_iterations': len(self.history)
        }
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.history, 'b-', linewidth=2, label='Objective Value')
            plt.axvline(x=self.best_iteration, color='r', linestyle='--', 
                       label=f'Best at iteration {self.best_iteration}')
            
            if self.should_stop:
                plt.axvline(x=self.stopped_iteration, color='orange', linestyle='--',
                           label=f'Early stopped at iteration {self.stopped_iteration}')
            
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Optimization History with Early Stopping')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"优化历史图已保存: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制历史图")

class AdaptiveEarlyStopping(EarlyStopping):
    """自适应早停机制"""
    
    def __init__(self, initial_patience: int = 5, min_delta: float = 0.01, 
                 patience_factor: float = 1.5, max_patience: int = 20):
        """
        初始化自适应早停机制
        
        Args:
            initial_patience: 初始耐心值
            min_delta: 最小改善阈值
            patience_factor: 耐心增长因子
            max_patience: 最大耐心值
        """
        super().__init__(initial_patience, min_delta)
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.improvement_count = 0
    
    def __call__(self, current_value: float, current_params: Optional[dict] = None) -> bool:
        """
        自适应早停检查
        
        Args:
            current_value: 当前目标值
            current_params: 当前参数（可选）
            
        Returns:
            是否应该停止训练
        """
        self.history.append(current_value)
        
        # 检查是否有改善
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.best_iteration = len(self.history) - 1
            if current_params is not None:
                self.best_params = current_params.copy()
            
            self.improvement_count += 1
            self.wait = 0
            
            # 自适应调整耐心值
            if self.improvement_count % 3 == 0:  # 每3次改善增加耐心
                old_patience = self.patience
                self.patience = min(
                    int(self.patience * self.patience_factor), 
                    self.max_patience
                )
                logger.info(f"耐心值从 {old_patience} 增加到 {self.patience}")
            
            logger.debug(f"发现改善: {current_value:.6f}")
        else:
            self.wait += 1
            logger.debug(f"无改善，等待计数: {self.wait}/{self.patience}")
        
        # 检查是否应该停止
        if self.wait >= self.patience:
            self.stopped_iteration = len(self.history) - 1
            self.should_stop = True
            logger.info(f"自适应早停触发：在第{self.stopped_iteration}次迭代，最佳值: {self.best_value:.6f}")
            return True
        
        return False

class ConvergenceDetector:
    """收敛检测器"""
    
    def __init__(self, window_size: int = 10, tolerance: float = 1e-6):
        """
        初始化收敛检测器
        
        Args:
            window_size: 滑动窗口大小
            tolerance: 收敛容忍度
        """
        self.window_size = window_size
        self.tolerance = tolerance
        self.values: List[float] = []
    
    def add_value(self, value: float) -> bool:
        """
        添加新值并检查收敛
        
        Args:
            value: 新的目标值
            
        Returns:
            是否已收敛
        """
        self.values.append(value)
        
        # 保持窗口大小
        if len(self.values) > self.window_size:
            self.values.pop(0)
        
        # 检查收敛
        if len(self.values) >= self.window_size:
            recent_values = np.array(self.values[-self.window_size:])
            
            # 计算方差
            variance = np.var(recent_values)
            
            # 计算相对变化
            if len(self.values) > 1:
                relative_change = abs(self.values[-1] - self.values[-2]) / (abs(self.values[-1]) + 1e-8)
            else:
                relative_change = float('inf')
            
            # 判断收敛
            converged = variance < self.tolerance and relative_change < self.tolerance
            
            if converged:
                logger.info(f"检测到收敛: 方差={variance:.2e}, 相对变化={relative_change:.2e}")
            
            return converged
        
        return False
    
    def get_trend(self) -> str:
        """
        获取当前趋势
        
        Returns:
            趋势描述字符串
        """
        if len(self.values) < 3:
            return "insufficient_data"
        
        recent_slope = np.polyfit(range(len(self.values)), self.values, 1)[0]
        
        if recent_slope < -self.tolerance:
            return "improving"
        elif recent_slope > self.tolerance:
            return "deteriorating"
        else:
            return "stable"

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
            min_delta=config.min_delta
        )
    else:
        return EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )