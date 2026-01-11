# 新增或修改我时需要修改这个文件夹中的README.md文件
"""结构化输出工具 - LLM输出解析与校验"""
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError
import json


class StructuredOutputTool:
    """
    结构化输出工具
    
    职责：
    - 解析LLM结构化输出
    - 校验字段完整性
    - 重试策略
    """
    
    def __init__(self, schema: Type[BaseModel], max_retries: int = 3):
        """
        Args:
            schema: Pydantic模型定义
            max_retries: 最大重试次数
        """
        self.schema = schema
        self.max_retries = max_retries
    
    def parse(self, llm_output: str) -> tuple[Optional[BaseModel], Optional[str]]:
        """
        解析LLM输出
        
        Returns:
            (parsed_object, error_message)
        """
        try:
            # 尝试直接解析JSON
            if isinstance(llm_output, str):
                # 尝试提取JSON部分
                json_str = self._extract_json(llm_output)
                data = json.loads(json_str)
            else:
                data = llm_output
            
            # 使用Pydantic验证
            parsed = self.schema(**data)
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON解析失败: {str(e)}"
        except ValidationError as e:
            return None, f"字段验证失败: {str(e)}"
        except Exception as e:
            return None, f"解析错误: {str(e)}"
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON部分"""
        # 移除可能的markdown代码块标记
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:].strip()
        elif text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        
        # 尝试找到JSON对象
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return text[start:end]
        
        # 尝试找到JSON数组
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]
        
        # 如果整个文本看起来就是JSON，直接返回
        if text.strip().startswith('{') or text.strip().startswith('['):
            return text.strip()
        
        return text
    
    def validate_evidence(self, evidence_text: str, target_sentence: str) -> bool:
        """
        验证evidence是否可在target_sentence中定位
        
        Returns:
            True if evidence is found in target_sentence
        """
        return evidence_text.lower() in target_sentence.lower()

