"""配置管理模块"""
from pydantic import BaseModel
from typing import Literal


class ChunkingConfig(BaseModel):
    """分块策略配置"""
    strategy: Literal["fixed", "semantic"] = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50


class LLMConfig(BaseModel):
    """LLM 配置"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    request_timeout: float = 120.0


class CriticConfig(BaseModel):
    """Critic 模块配置"""
    threshold: float = 4.0  # 1-5 分，低于此值触发反思
    max_retries: int = 3


class AppConfig(BaseModel):
    """应用总配置"""
    chunking: ChunkingConfig = ChunkingConfig()
    llm: LLMConfig = LLMConfig()
    critic: CriticConfig = CriticConfig()
    embedding_model: str = "BAAI/bge-small-zh-v1.5"


# 默认配置实例
default_config = AppConfig()
