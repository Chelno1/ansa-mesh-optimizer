"""
执行器基类
Base Executor
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseExecutor(ABC):
    """执行器基类"""
    
    def __init__(self, task_id: str, params: Dict[str, Any]):
        self.task_id = task_id
        self.params = params
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """
        执行任务
        
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    async def validate(self) -> bool:
        """
        验证参数
        
        Returns:
            验证是否通过
        """
        pass
