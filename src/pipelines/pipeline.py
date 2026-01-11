# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Pipeline定义 - 执行顺序与依赖"""
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class PipelineStep:
    """流水线步骤"""
    
    def __init__(
        self,
        name: str,
        agent_class: type,
        dependencies: List[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Args:
            name: 步骤名称
            agent_class: Agent类
            dependencies: 依赖的步骤名称列表
            config: 步骤配置
        """
        self.name = name
        self.agent_class = agent_class
        self.dependencies = dependencies or []
        self.config = config or {}


class Pipeline:
    """
    Pipeline定义
    
    职责：
    - 定义执行顺序与依赖
    - 支持全流程运行
    - 支持断点续跑
    """
    
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
        self.step_map = {step.name: step for step in steps}
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序（拓扑排序）"""
        # 简单的拓扑排序实现
        visited = set()
        result = []
        
        def visit(step_name: str):
            if step_name in visited:
                return
            visited.add(step_name)
            step = self.step_map[step_name]
            for dep in step.dependencies:
                visit(dep)
            result.append(step_name)
        
        for step in self.steps:
            visit(step.name)
        
        return result
    
    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """获取步骤"""
        return self.step_map.get(step_name)
    
    def can_run_step(self, step_name: str, completed_steps: set) -> bool:
        """检查步骤是否可以运行"""
        step = self.get_step(step_name)
        if not step:
            return False
        return all(dep in completed_steps for dep in step.dependencies)

