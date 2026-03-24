"""src/reflective_agent.py - 自动反思学术代理 (核心模块)"""
from dataclasses import dataclass, field
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama

from src.config import AppConfig
from src.critic import Critic, CriticResult


@dataclass
class ChatResponse:
    """对话响应"""

    answer: str
    reflection_count: int  # 经历了几次反思
    final_score: float  # 最终质量分数
    sources: List[str]  # 引用的文档来源
    reflection_history: List[CriticResult] = field(default_factory=list)


class ReflectiveAgent:
    """自动反思学术代理

    核心工作流：
    1. 接收用户问题
    2. 检索相关文档块
    3. 生成初步回答
    4. Critic 评估回答质量
    5. 若不达标，根据反馈重写查询并重新生成
    6. 循环直至达标或达到最大重试次数
    """

    def __init__(self, index: VectorStoreIndex, config: AppConfig):
        self.config = config
        self.index = index

        # 初始化 LLM
        self.llm = Ollama(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            request_timeout=config.llm.request_timeout,
        )

        # 初始化对话记忆
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
        )

        # 初始化检索器
        self.retriever = index.as_retriever(similarity_top_k=5)

        # 初始化 Critic
        self.critic = Critic(config.critic, config.llm)

        # 初始化对话引擎
        self._init_chat_engine()

    def _init_chat_engine(self):
        """初始化对话引擎"""
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
        )

    def chat(self, user_message: str) -> ChatResponse:
        """处理用户消息，返回经过反思优化的回答"""
        current_query = user_message
        reflection_history = []
        answer = ""
        sources = []

        for attempt in range(self.config.critic.max_retries + 1):
            # 检索相关文档
            retrieved_nodes = self.retriever.retrieve(current_query)
            context = "\n\n".join([node.text for node in retrieved_nodes])
            sources = [
                node.metadata.get("file_name", "unknown") for node in retrieved_nodes
            ]

            # 生成回答
            response = self.chat_engine.chat(current_query)
            answer = str(response)

            # Critic 评估
            critic_result = self.critic.evaluate(
                question=user_message,  # 始终用原始问题评估
                context=context,
                answer=answer,
            )
            reflection_history.append(critic_result)

            # 检查是否通过
            if not critic_result.needs_retry:
                return ChatResponse(
                    answer=answer,
                    reflection_count=attempt,
                    final_score=critic_result.score,
                    sources=list(set(sources)),
                    reflection_history=reflection_history,
                )

            # 需要重试：使用建议的新查询或保持原查询
            if critic_result.suggested_query:
                current_query = critic_result.suggested_query

        # 达到最大重试次数，返回最后一次的结果
        return ChatResponse(
            answer=answer,
            reflection_count=self.config.critic.max_retries,
            final_score=reflection_history[-1].score if reflection_history else 0.0,
            sources=list(set(sources)),
            reflection_history=reflection_history,
        )

    def reset_memory(self):
        """重置对话记忆（开始新会话）"""
        self.memory.reset()
        self._init_chat_engine()
