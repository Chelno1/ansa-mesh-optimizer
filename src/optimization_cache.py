#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化缓存管理模块

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import pickle
import hashlib
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import json
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OptimizationCache:
    """优化结果缓存管理器"""
    
    def __init__(self, cache_file: str = 'optimization_cache.pkl', max_age_days: int = 30):
        """
        初始化缓存管理器
        
        Args:
            cache_file: 缓存文件路径
            max_age_days: 缓存最大存活天数
        """
        self.cache_file = Path(cache_file)
        self.max_age = timedelta(days=max_age_days)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        
        self._load_cache()
    
    def _compute_hash(self, params: Dict[str, float]) -> str:
        """
        计算参数哈希值
        
        Args:
            params: 参数字典
            
        Returns:
            参数的哈希值
        """
        # 确保参数的一致性（排序并转换为字符串）
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, params: Dict[str, float]) -> Optional[float]:
        """
        从缓存获取结果
        
        Args:
            params: 参数字典
            
        Returns:
            缓存的结果值，如果不存在则返回None
        """
        params_hash = self._compute_hash(params)
        
        if params_hash in self.cache:
            entry = self.cache[params_hash]
            
            # 检查缓存是否过期
            if self._is_cache_valid(entry):
                self.hits += 1
                logger.debug(f"缓存命中: {params_hash}")
                return entry['result']
            else:
                # 删除过期缓存
                del self.cache[params_hash]
                logger.debug(f"删除过期缓存: {params_hash}")
        
        self.misses += 1
        logger.debug(f"缓存未命中: {params_hash}")
        return None
    
    def set(self, params: Dict[str, float], result: float) -> None:
        """
        设置缓存
        
        Args:
            params: 参数字典
            result: 结果值
        """
        params_hash = self._compute_hash(params)
        
        self.cache[params_hash] = {
            'params': params.copy(),
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'evaluation_time': time.time()
        }
        
        logger.debug(f"设置缓存: {params_hash} -> {result}")
        
        # 定期保存缓存
        if len(self.cache) % 10 == 0:  # 每10个条目保存一次
            self._save_cache()
    
    def _is_cache_valid(self, entry: Dict[str, Any]) -> bool:
        """
        检查缓存条目是否有效
        
        Args:
            entry: 缓存条目
            
        Returns:
            缓存是否有效
        """
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            return datetime.now() - timestamp < self.max_age
        except (KeyError, ValueError):
            return False
    
    def _load_cache(self) -> None:
        """从文件加载缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.hits = data.get('hits', 0)
                    self.misses = data.get('misses', 0)
                
                # 清理过期缓存
                self._cleanup_expired()
                
                logger.info(f"加载缓存: {len(self.cache)} 个条目")
            else:
                logger.info("缓存文件不存在，创建新缓存")
                
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """保存缓存到文件"""
        try:
            # 确保目录存在
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 清理过期缓存
            self._cleanup_expired()
            
            data = {
                'cache': self.cache,
                'hits': self.hits,
                'misses': self.misses,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"缓存已保存: {len(self.cache)} 个条目")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        expired_keys = []
        
        for key, entry in self.cache.items():
            if not self._is_cache_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_file': str(self.cache_file),
            'file_size_mb': self.cache_file.stat().st_size / 1024 / 1024 if self.cache_file.exists() else 0
        }
    
    def export_cache_data(self, export_file: str) -> None:
        """
        导出缓存数据到JSON文件
        
        Args:
            export_file: 导出文件路径
        """
        try:
            export_data = []
            
            for params_hash, entry in self.cache.items():
                export_data.append({
                    'hash': params_hash,
                    'params': entry['params'],
                    'result': entry['result'],
                    'timestamp': entry['timestamp']
                })
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"缓存数据已导出到: {export_file}")
            
        except Exception as e:
            logger.error(f"导出缓存数据失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动保存缓存"""
        self._save_cache()

class CachedEvaluator:
    """带缓存的评估器装饰器"""
    
    def __init__(self, evaluator, cache: OptimizationCache):
        """
        初始化缓存评估器
        
        Args:
            evaluator: 原始评估器
            cache: 缓存管理器
        """
        self.evaluator = evaluator
        self.cache = cache
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        带缓存的网格评估
        
        Args:
            params: 参数字典
            
        Returns:
            评估结果
        """
        # 首先尝试从缓存获取
        cached_result = self.cache.get(params)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，执行实际评估
        result = self.evaluator.evaluate_mesh(params)
        
        # 将结果存入缓存
        if result != float('inf'):  # 只缓存有效结果
            self.cache.set(params, result)
        
        return result
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """参数验证（直接委托给原评估器）"""
        return self.evaluator.validate_params(params)
    
    def __getattr__(self, name):
        """委托其他属性访问给原评估器"""
        return getattr(self.evaluator, name)