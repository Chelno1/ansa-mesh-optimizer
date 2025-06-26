#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化缓存管理模块 - 增强版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 内存管理，性能优化，数据完整性
"""

import pickle
import hashlib
import logging
import sqlite3
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import json
import time
import threading
from datetime import datetime, timedelta
from collections import OrderedDict
import gzip
import tempfile

logger = logging.getLogger(__name__)

def normalize_for_json(obj):
    """标准化对象以便JSON序列化 - 增强版本"""
    if isinstance(obj, dict):
        return {key: normalize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_for_json(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy类型
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return float(obj)
    elif hasattr(obj, 'dtype'):  # numpy数组等
        if hasattr(obj, 'size') and obj.size == 1:
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return float(obj)
        else:
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'isoformat'):  # datetime对象
        return obj.isoformat()
    else:
        return str(obj)

class CacheEntry:
    """缓存条目类"""
    
    __slots__ = ['params_hash', 'params', 'result', 'timestamp', 'access_count', 'evaluation_time']
    
    def __init__(self, params_hash: str, params: Dict[str, Any], result: float, 
                 timestamp: str = None, evaluation_time: float = None):
        self.params_hash = params_hash
        self.params = params
        self.result = result
        self.timestamp = timestamp or datetime.now().isoformat()
        self.access_count = 1
        self.evaluation_time = evaluation_time or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'params_hash': self.params_hash,
            'params': self.params,
            'result': self.result,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'evaluation_time': self.evaluation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建"""
        entry = cls(
            params_hash=data['params_hash'],
            params=data['params'],
            result=data['result'],
            timestamp=data.get('timestamp'),
            evaluation_time=data.get('evaluation_time')
        )
        entry.access_count = data.get('access_count', 1)
        return entry

class OptimizationCache:
    """优化结果缓存管理器 - 增强版本"""
    
    def __init__(self, cache_file: str = 'optimization_cache.pkl', 
                 max_age_days: int = 30,
                 max_entries: int = 10000,
                 use_compression: bool = True,
                 use_database: bool = False):
        """
        初始化缓存管理器
        
        Args:
            cache_file: 缓存文件路径
            max_age_days: 缓存最大存活天数
            max_entries: 最大缓存条目数
            use_compression: 是否使用压缩
            use_database: 是否使用数据库存储
        """
        self.cache_file = Path(cache_file)
        self.max_age = timedelta(days=max_age_days)
        self.max_entries = max_entries
        self.use_compression = use_compression
        self.use_database = use_database
        
        # 缓存存储
        if use_database:
            self.db_file = self.cache_file.with_suffix('.db')
            self._init_database()
            self.cache = {}  # 内存缓存用于快速访问
        else:
            self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 自动保存配置
        self._auto_save_threshold = 50  # 每50次操作自动保存一次
        self._operations_since_save = 0
        
        # 内存管理
        self._memory_threshold = 100 * 1024 * 1024  # 100MB
        self._check_memory_usage = True
        
        self._load_cache()
    
    def _init_database(self) -> None:
        """初始化SQLite数据库"""
        try:
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.db_file)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        params_hash TEXT PRIMARY KEY,
                        params TEXT NOT NULL,
                        result REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        access_count INTEGER DEFAULT 1,
                        evaluation_time REAL
                    )
                ''')
                
                # 创建索引
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)')
                
                conn.commit()
            
            logger.info(f"数据库缓存初始化完成: {self.db_file}")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            self.use_database = False
    
    def _compute_hash(self, params: Dict[str, float]) -> str:
        """
        计算参数哈希值 - 改进版本
        
        Args:
            params: 参数字典
            
        Returns:
            参数的哈希值
        """
        try:
            # 转换numpy类型为Python原生类型
            normalized_params = normalize_for_json(params)
            
            # 确保参数的一致性（排序并转换为字符串）
            sorted_params = sorted(normalized_params.items())
            param_str = json.dumps(sorted_params, sort_keys=True, separators=(',', ':'))
            
            # 使用SHA256而不是MD5以获得更好的散列性质
            return hashlib.sha256(param_str.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"计算参数哈希失败: {e}")
            # 降级到简单哈希
            return hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
    
    def get(self, params: Dict[str, float]) -> Optional[float]:
        """
        从缓存获取结果 - 线程安全版本
        
        Args:
            params: 参数字典
            
        Returns:
            缓存的结果值，如果不存在则返回None
        """
        with self._lock:
            self.total_requests += 1
            params_hash = self._compute_hash(params)
            
            try:
                if self.use_database:
                    entry = self._get_from_database(params_hash)
                else:
                    entry = self.cache.get(params_hash)
                
                if entry and self._is_cache_valid(entry):
                    # 更新访问统计
                    entry.access_count += 1
                    
                    # 在OrderedDict中移动到末尾（LRU）
                    if not self.use_database and params_hash in self.cache:
                        self.cache.move_to_end(params_hash)
                    
                    self.hits += 1
                    logger.debug(f"缓存命中: {params_hash[:8]}...")
                    return entry.result
                    
                elif entry:
                    # 删除过期缓存
                    self._remove_entry(params_hash)
                    logger.debug(f"删除过期缓存: {params_hash[:8]}...")
                
                self.misses += 1
                logger.debug(f"缓存未命中: {params_hash[:8]}...")
                return None
                
            except Exception as e:
                logger.error(f"缓存获取失败: {e}")
                self.misses += 1
                return None
    
    def set(self, params: Dict[str, float], result: float) -> None:
        """
        设置缓存 - 线程安全版本
        
        Args:
            params: 参数字典
            result: 结果值
        """
        with self._lock:
            try:
                params_hash = self._compute_hash(params)
                
                # 转换numpy类型为Python原生类型
                normalized_params = normalize_for_json(params)
                normalized_result = normalize_for_json(result)
                
                # 创建缓存条目
                entry = CacheEntry(
                    params_hash=params_hash,
                    params=normalized_params,
                    result=float(normalized_result),
                    timestamp=datetime.now().isoformat(),
                    evaluation_time=time.time()
                )
                
                if self.use_database:
                    self._set_to_database(entry)
                    # 同时保存到内存缓存以提高访问速度
                    self.cache[params_hash] = entry
                    
                    # 限制内存缓存大小
                    if len(self.cache) > 1000:
                        # 移除最旧的条目
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                else:
                    self.cache[params_hash] = entry
                    
                    # LRU淘汰策略
                    if len(self.cache) > self.max_entries:
                        # 移除最旧的条目
                        oldest_key, _ = self.cache.popitem(last=False)
                        logger.debug(f"LRU淘汰缓存条目: {oldest_key[:8]}...")
                
                logger.debug(f"设置缓存: {params_hash[:8]}... -> {normalized_result}")
                
                # 自动保存检查
                self._operations_since_save += 1
                if self._operations_since_save >= self._auto_save_threshold:
                    self._auto_save()
                
                # 内存使用检查
                if self._check_memory_usage:
                    self._check_and_manage_memory()
                
            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
    
    def _get_from_database(self, params_hash: str) -> Optional[CacheEntry]:
        """从数据库获取缓存条目"""
        try:
            with sqlite3.connect(str(self.db_file)) as conn:
                cursor = conn.execute(
                    'SELECT params_hash, params, result, timestamp, access_count, evaluation_time '
                    'FROM cache_entries WHERE params_hash = ?',
                    (params_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    params_data = json.loads(row[1])
                    entry = CacheEntry(
                        params_hash=row[0],
                        params=params_data,
                        result=row[2],
                        timestamp=row[3],
                        evaluation_time=row[5]
                    )
                    entry.access_count = row[4]
                    
                    # 更新访问计数
                    conn.execute(
                        'UPDATE cache_entries SET access_count = access_count + 1 WHERE params_hash = ?',
                        (params_hash,)
                    )
                    conn.commit()
                    
                    return entry
                
                return None
                
        except Exception as e:
            logger.error(f"数据库查询失败: {e}")
            return None
    
    def _set_to_database(self, entry: CacheEntry) -> None:
        """设置缓存条目到数据库"""
        try:
            with sqlite3.connect(str(self.db_file)) as conn:
                params_json = json.dumps(entry.params, separators=(',', ':'))
                
                conn.execute(
                    'INSERT OR REPLACE INTO cache_entries '
                    '(params_hash, params, result, timestamp, access_count, evaluation_time) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (entry.params_hash, params_json, entry.result, 
                     entry.timestamp, entry.access_count, entry.evaluation_time)
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"数据库写入失败: {e}")
    
    def _remove_entry(self, params_hash: str) -> None:
        """移除缓存条目"""
        try:
            if self.use_database:
                with sqlite3.connect(str(self.db_file)) as conn:
                    conn.execute('DELETE FROM cache_entries WHERE params_hash = ?', (params_hash,))
                    conn.commit()
            
            if params_hash in self.cache:
                del self.cache[params_hash]
                
        except Exception as e:
            logger.error(f"移除缓存条目失败: {e}")
    
    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """
        检查缓存条目是否有效
        
        Args:
            entry: 缓存条目
            
        Returns:
            缓存是否有效
        """
        try:
            timestamp = datetime.fromisoformat(entry.timestamp)
            age = datetime.now() - timestamp
            return age < self.max_age
        except (ValueError, TypeError):
            return False
    
    def _load_cache(self) -> None:
        """从文件加载缓存"""
        try:
            if self.use_database:
                self._load_from_database()
            else:
                self._load_from_file()
            
            # 清理过期缓存
            self._cleanup_expired()
            
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            if not self.use_database:
                self.cache = OrderedDict()
    
    def _load_from_database(self) -> None:
        """从数据库加载缓存"""
        if not self.db_file.exists():
            logger.info("数据库缓存文件不存在，创建新缓存")
            return
        
        try:
            with sqlite3.connect(str(self.db_file)) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
                count = cursor.fetchone()[0]
                logger.info(f"从数据库加载缓存: {count} 个条目")
                
                # 加载最近访问的条目到内存
                cursor = conn.execute(
                    'SELECT params_hash, params, result, timestamp, access_count, evaluation_time '
                    'FROM cache_entries ORDER BY access_count DESC LIMIT 1000'
                )
                
                for row in cursor:
                    params_data = json.loads(row[1])
                    entry = CacheEntry(
                        params_hash=row[0],
                        params=params_data,
                        result=row[2],
                        timestamp=row[3],
                        evaluation_time=row[5]
                    )
                    entry.access_count = row[4]
                    self.cache[entry.params_hash] = entry
                
        except Exception as e:
            logger.error(f"从数据库加载缓存失败: {e}")
    
    def _load_from_file(self) -> None:
        """从文件加载缓存"""
        if not self.cache_file.exists():
            logger.info("缓存文件不存在，创建新缓存")
            return
        
        try:
            # 尝试不同的加载方式
            if self.use_compression and self.cache_file.suffix == '.gz':
                with gzip.open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            # 兼容性处理
            if isinstance(data, dict):
                if 'cache' in data:
                    # 新格式
                    cache_data = data['cache']
                    self.hits = data.get('hits', 0)
                    self.misses = data.get('misses', 0)
                    self.total_requests = data.get('total_requests', 0)
                else:
                    # 旧格式
                    cache_data = data
                
                # 转换为CacheEntry对象
                self.cache = OrderedDict()
                for params_hash, entry_data in cache_data.items():
                    if isinstance(entry_data, dict):
                        if 'params' in entry_data:
                            # 新格式CacheEntry
                            entry = CacheEntry.from_dict(entry_data)
                        else:
                            # 旧格式，转换为新格式
                            entry = CacheEntry(
                                params_hash=params_hash,
                                params=entry_data.get('params', {}),
                                result=entry_data.get('result', 0.0),
                                timestamp=entry_data.get('timestamp', datetime.now().isoformat()),
                                evaluation_time=entry_data.get('evaluation_time', time.time())
                            )
                    else:
                        # 非常旧的格式，跳过
                        continue
                    
                    self.cache[params_hash] = entry
            
            logger.info(f"加载缓存: {len(self.cache)} 个条目")
            
        except Exception as e:
            logger.error(f"加载缓存文件失败: {e}")
            self.cache = OrderedDict()
    
    def _save_cache(self) -> None:
        """保存缓存到文件"""
        try:
            if self.use_database:
                # 数据库模式下只需要同步内存统计
                self._save_statistics()
            else:
                self._save_to_file()
            
            self._operations_since_save = 0
            logger.debug(f"缓存已保存: {len(self.cache)} 个条目")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _save_to_file(self) -> None:
        """保存缓存到文件"""
        # 确保目录存在
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 清理过期缓存
        self._cleanup_expired()
        
        # 准备保存数据
        data = {
            'cache': {k: v.to_dict() for k, v in self.cache.items()},
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'saved_at': datetime.now().isoformat(),
            'version': '1.2.0'
        }
        
        # 选择保存格式
        if self.use_compression:
            cache_file = self.cache_file.with_suffix('.gz')
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_statistics(self) -> None:
        """保存统计信息"""
        stats_file = self.cache_file.with_suffix('.stats.json')
        try:
            stats = {
                'hits': self.hits,
                'misses': self.misses,
                'total_requests': self.total_requests,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.warning(f"保存统计信息失败: {e}")
    
    def _auto_save(self) -> None:
        """自动保存"""
        try:
            self._save_cache()
        except Exception as e:
            logger.warning(f"自动保存失败: {e}")
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        expired_keys = []
        
        if self.use_database:
            try:
                with sqlite3.connect(str(self.db_file)) as conn:
                    cutoff_time = (datetime.now() - self.max_age).isoformat()
                    cursor = conn.execute(
                        'SELECT params_hash FROM cache_entries WHERE timestamp < ?',
                        (cutoff_time,)
                    )
                    expired_keys = [row[0] for row in cursor]
                    
                    if expired_keys:
                        placeholders = ','.join('?' * len(expired_keys))
                        conn.execute(
                            f'DELETE FROM cache_entries WHERE params_hash IN ({placeholders})',
                            expired_keys
                        )
                        conn.commit()
                        
            except Exception as e:
                logger.error(f"数据库清理失败: {e}")
        
        # 清理内存缓存
        for key, entry in list(self.cache.items()):
            if not self._is_cache_valid(entry):
                expired_keys.append(key)
                del self.cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _check_and_manage_memory(self) -> None:
        """检查和管理内存使用"""
        try:
            import sys
            
            # 简单的内存检查
            cache_size = sys.getsizeof(self.cache)
            for entry in self.cache.values():
                cache_size += sys.getsizeof(entry.params) + sys.getsizeof(entry.result)
            
            if cache_size > self._memory_threshold:
                # 清理最少使用的条目
                sorted_items = sorted(self.cache.items(), 
                                    key=lambda x: x[1].access_count)
                
                # 移除访问次数最少的20%条目
                remove_count = max(1, len(self.cache) // 5)
                for i in range(remove_count):
                    if sorted_items:
                        key, _ = sorted_items.pop(0)
                        if key in self.cache:
                            del self.cache[key]
                
                logger.info(f"内存管理: 移除了 {remove_count} 个低访问缓存条目")
                
        except Exception as e:
            logger.warning(f"内存管理检查失败: {e}")
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            if self.use_database:
                try:
                    with sqlite3.connect(str(self.db_file)) as conn:
                        conn.execute('DELETE FROM cache_entries')
                        conn.commit()
                except Exception as e:
                    logger.error(f"清空数据库缓存失败: {e}")
            
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.total_requests = 0
            
            if self.cache_file.exists():
                self.cache_file.unlink()
            
            logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0
        
        stats = {
            'total_entries': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate': hit_rate,
            'cache_file': str(self.cache_file),
            'use_database': self.use_database,
            'use_compression': self.use_compression,
            'max_entries': self.max_entries,
            'max_age_days': self.max_age.days
        }
        
        # 文件大小信息
        try:
            if self.use_database and self.db_file.exists():
                stats['file_size_mb'] = self.db_file.stat().st_size / 1024 / 1024
            elif self.cache_file.exists():
                stats['file_size_mb'] = self.cache_file.stat().st_size / 1024 / 1024
            else:
                stats['file_size_mb'] = 0
        except Exception:
            stats['file_size_mb'] = 0
        
        # 数据库特定统计
        if self.use_database:
            try:
                with sqlite3.connect(str(self.db_file)) as conn:
                    cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
                    stats['database_entries'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute('SELECT AVG(access_count) FROM cache_entries')
                    avg_access = cursor.fetchone()[0]
                    stats['avg_access_count'] = avg_access if avg_access else 0
                    
            except Exception as e:
                logger.warning(f"获取数据库统计失败: {e}")
        
        return stats
    
    def export_cache_data(self, export_file: str, format: str = 'json') -> None:
        """
        导出缓存数据到文件
        
        Args:
            export_file: 导出文件路径
            format: 导出格式 ('json', 'csv')
        """
        try:
            export_path = Path(export_file)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                self._export_to_json(export_path)
            elif format.lower() == 'csv':
                self._export_to_csv(export_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"缓存数据已导出到: {export_file}")
            
        except Exception as e:
            logger.error(f"导出缓存数据失败: {e}")
    
    def _export_to_json(self, export_path: Path) -> None:
        """导出为JSON格式"""
        export_data = []
        
        if self.use_database:
            with sqlite3.connect(str(self.db_file)) as conn:
                cursor = conn.execute(
                    'SELECT params_hash, params, result, timestamp, access_count, evaluation_time '
                    'FROM cache_entries ORDER BY timestamp DESC'
                )
                
                for row in cursor:
                    params_data = json.loads(row[1])
                    export_data.append({
                        'hash': row[0],
                        'params': params_data,
                        'result': row[2],
                        'timestamp': row[3],
                        'access_count': row[4],
                        'evaluation_time': row[5]
                    })
        else:
            for entry in self.cache.values():
                export_data.append({
                    'hash': entry.params_hash,
                    'params': entry.params,
                    'result': entry.result,
                    'timestamp': entry.timestamp,
                    'access_count': entry.access_count,
                    'evaluation_time': entry.evaluation_time
                })
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_entries': len(export_data),
                    'cache_stats': self.get_stats()
                },
                'entries': export_data
            }, f, indent=2, ensure_ascii=False)
    
    def _export_to_csv(self, export_path: Path) -> None:
        """导出为CSV格式"""
        try:
            import csv
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入标题行
                writer.writerow(['params_hash', 'params_json', 'result', 'timestamp', 'access_count', 'evaluation_time'])
                
                if self.use_database:
                    with sqlite3.connect(str(self.db_file)) as conn:
                        cursor = conn.execute(
                            'SELECT params_hash, params, result, timestamp, access_count, evaluation_time '
                            'FROM cache_entries ORDER BY timestamp DESC'
                        )
                        
                        for row in cursor:
                            writer.writerow(row)
                else:
                    for entry in self.cache.values():
                        writer.writerow([
                            entry.params_hash,
                            json.dumps(entry.params, separators=(',', ':')),
                            entry.result,
                            entry.timestamp,
                            entry.access_count,
                            entry.evaluation_time
                        ])
                        
        except ImportError:
            logger.error("CSV导出需要csv模块")
        except Exception as e:
            logger.error(f"CSV导出失败: {e}")
    
    def optimize_cache(self) -> Dict[str, int]:
        """优化缓存性能"""
        with self._lock:
            operations = {
                'expired_removed': 0,
                'duplicates_removed': 0,
                'total_before': len(self.cache)
            }
            
            # 清理过期条目
            before_count = len(self.cache)
            self._cleanup_expired()
            operations['expired_removed'] = before_count - len(self.cache)
            
            # 清理重复条目（基于参数相似性）
            if not self.use_database:
                operations['duplicates_removed'] = self._remove_similar_entries()
            
            # 数据库优化
            if self.use_database:
                try:
                    with sqlite3.connect(str(self.db_file)) as conn:
                        conn.execute('VACUUM')
                        conn.execute('ANALYZE')
                        conn.commit()
                except Exception as e:
                    logger.warning(f"数据库优化失败: {e}")
            
            operations['total_after'] = len(self.cache)
            
            logger.info(f"缓存优化完成: {operations}")
            return operations
    
    def _remove_similar_entries(self) -> int:
        """移除相似的缓存条目"""
        # 这里可以实现更复杂的相似性检测
        # 暂时返回0，表示没有移除任何条目
        return 0
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动保存缓存"""
        self._save_cache()

class CachedEvaluator:
    """带缓存的评估器装饰器 - 增强版本"""
    
    def __init__(self, evaluator, cache: OptimizationCache):
        """
        初始化缓存评估器
        
        Args:
            evaluator: 原始评估器
            cache: 缓存管理器
        """
        self.evaluator = evaluator
        self.cache = cache
        self._evaluation_count = 0
        self._cache_hits = 0
        self._start_time = time.time()
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        带缓存的网格评估
        
        Args:
            params: 参数字典
            
        Returns:
            评估结果
        """
        self._evaluation_count += 1
        
        # 首先尝试从缓存获取
        cached_result = self.cache.get(params)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result
        
        # 缓存未命中，执行实际评估
        start_time = time.time()
        try:
            result = self.evaluator.evaluate_mesh(params)
            evaluation_time = time.time() - start_time
            
            # 将结果存入缓存（只缓存有效结果）
            if result != float('inf') and not (result is None):
                self.cache.set(params, result)
            
            # 记录评估时间
            if hasattr(self, '_total_evaluation_time'):
                self._total_evaluation_time += evaluation_time
            else:
                self._total_evaluation_time = evaluation_time
            
            return result
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return float('inf')
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """参数验证（直接委托给原评估器）"""
        return self.evaluator.validate_params(params)
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        total_time = time.time() - self._start_time
        cache_hit_rate = self._cache_hits / self._evaluation_count if self._evaluation_count > 0 else 0
        
        stats = {
            'total_evaluations': self._evaluation_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._evaluation_count - self._cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'total_runtime': total_time,
            'avg_evaluation_time': getattr(self, '_total_evaluation_time', 0) / max(1, self._evaluation_count - self._cache_hits)
        }
        
        return stats
    
    def __getattr__(self, name):
        """委托其他属性访问给原评估器"""
        return getattr(self.evaluator, name)

# 工厂函数
def create_cache(cache_type: str = 'file', **kwargs) -> OptimizationCache:
    """
    创建缓存实例的工厂函数
    
    Args:
        cache_type: 缓存类型 ('file', 'database', 'memory')
        **kwargs: 缓存配置参数
        
    Returns:
        配置好的缓存实例
    """
    if cache_type == 'database':
        kwargs['use_database'] = True
    elif cache_type == 'memory':
        kwargs['max_age_days'] = 1  # 内存缓存只保留1天
        kwargs['max_entries'] = 1000
    
    return OptimizationCache(**kwargs)

if __name__ == "__main__":
    # 测试缓存功能
    print("=== 缓存系统测试 ===")
    
    # 创建测试缓存
    cache = OptimizationCache(
        cache_file='test_cache.pkl',
        max_entries=10,
        use_compression=True
    )
    
    # 测试基本操作
    test_params = {'x': 1.0, 'y': 2.0, 'z': 3.0}
    test_result = 42.0
    
    print(f"设置缓存: {test_params} -> {test_result}")
    cache.set(test_params, test_result)
    
    print(f"获取缓存: {cache.get(test_params)}")
    print(f"获取不存在的缓存: {cache.get({'a': 1, 'b': 2})}")
    
    # 测试统计信息
    stats = cache.get_stats()
    print(f"缓存统计: {stats}")
    
    # 测试导出
    cache.export_cache_data('test_export.json', 'json')
    
    # 清理
    cache.clear()
    
    print("缓存系统测试完成!")