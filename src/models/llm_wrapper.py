# 新增或修改我时需要修改这个文件夹中的README.md文件
"""LLM封装 - 支持结构化输出"""
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None


class LLMWrapper:
    """
    LLM封装
    
    职责：
    - 统一LLM接口
    - 支持Function Calling / Tool Call
    - 支持结构化输出（Ollama JSON格式）
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_reasoning: bool = False
    ):
        """
        Args:
            provider: 提供商（openai/ollama）
            model_name: 模型名称
            temperature: 温度参数
            api_key: API密钥（OpenAI需要）
            base_url: API基础URL（Ollama需要，默认http://localhost:11434）
            enable_reasoning: 是否启用reasoning模式（默认False）
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.enable_reasoning = enable_reasoning
        
        if provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain-openai is required for OpenAI provider")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                reasoning=enable_reasoning
            )
        elif provider == "ollama":
            if ChatOllama is None:
                raise ImportError("langchain-ollama is required for Ollama provider")
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url or "http://localhost:11434",
                reasoning=enable_reasoning
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: openai, ollama")
    
    def invoke_with_tools(
        self,
        prompt: str,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None
    ) -> Any:
        """
        使用工具调用（Function Calling）
        
        Args:
            prompt: 提示词
            tools: 工具列表
            tool_choice: 工具选择策略
        
        Returns:
            LLM响应
        """
        if tools:
            return self.llm.bind_tools(tools, tool_choice=tool_choice).invoke(prompt)
        return self.llm.invoke(prompt)
    
    def invoke_structured(
        self,
        prompt: str,
        schema: Type[BaseModel]
    ) -> Any:
        """
        结构化输出
        
        Args:
            prompt: 提示词
            schema: Pydantic模型
        
        Returns:
            LLM响应（需要后续解析）
        """
        if self.provider == "ollama":
            # Ollama 使用 format="json" 参数强制 JSON 输出
            # 需要在 prompt 中明确要求 JSON 格式
            json_prompt = self._add_json_format_instruction(prompt, schema)
            
            # 尝试使用 bind 方法设置 format 参数（如果支持）
            try:
                # LangChain ChatOllama 可能支持通过 bind_kwargs 传递额外参数
                if hasattr(self.llm, 'bind'):
                    # 尝试绑定 format 参数
                    try:
                        bound_llm = self.llm.bind(format="json")
                        response = bound_llm.invoke(json_prompt)
                    except (TypeError, AttributeError):
                        # 如果不支持 format 参数，回退到普通调用
                        response = self.llm.invoke(json_prompt)
                else:
                    response = self.llm.invoke(json_prompt)
            except Exception:
                # 如果出错，回退到普通调用
                response = self.llm.invoke(json_prompt)
            return response
        elif self.provider == "openai":
            # OpenAI 使用 with_structured_output（如果支持）
            if hasattr(self.llm, 'with_structured_output'):
                return self.llm.with_structured_output(schema).invoke(prompt)
            else:
                # 回退到普通调用+解析
                response = self.llm.invoke(prompt)
                return response
        else:
            # 默认处理
            response = self.llm.invoke(prompt)
            return response
    
    def _add_json_format_instruction(self, prompt: str, schema: Type[BaseModel]) -> str:
        """
        为 Ollama 添加 JSON 格式指令
        
        Args:
            prompt: 原始提示词
            schema: Pydantic模型（用于生成JSON Schema）
        
        Returns:
            增强后的提示词
        """
        # 生成 JSON Schema 描述
        schema_dict = schema.model_json_schema()
        
        # 添加 JSON 格式要求
        example_json = self._schema_to_example_json(schema_dict)
        json_instruction = f"""

**重要：请严格按照以下 JSON 格式输出，不要包含任何其他文本、解释或markdown代码块标记。**

输出格式示例：
{example_json}

请确保：
1. 输出是纯 JSON 格式，不包含 ```json 或 ``` 标记
2. 所有字段都必须符合上述结构
3. 如果某个字段没有值，使用 null 或空字符串
4. 直接输出 JSON，不要有任何前缀或后缀文本
"""
        
        return prompt + json_instruction
    
    def _schema_to_example_json(self, schema_dict: dict) -> str:
        """
        将 JSON Schema 转换为示例 JSON（简化版）
        
        Args:
            schema_dict: JSON Schema 字典
        
        Returns:
            示例 JSON 字符串
        """
        # 简化实现：返回一个基本的 JSON 结构示例
        # 实际可以根据 schema 生成更准确的示例
        if "properties" in schema_dict:
            example = {}
            for key, value in schema_dict["properties"].items():
                if "type" in value:
                    if value["type"] == "string":
                        example[key] = ""
                    elif value["type"] == "integer":
                        example[key] = 0
                    elif value["type"] == "number":
                        example[key] = 0.0
                    elif value["type"] == "boolean":
                        example[key] = False
                    elif value["type"] == "array":
                        example[key] = []
                    elif value["type"] == "object":
                        example[key] = {}
                else:
                    example[key] = None
            import json
            return json.dumps(example, indent=2, ensure_ascii=False)
        return "{}"

