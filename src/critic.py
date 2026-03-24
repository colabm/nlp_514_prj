"""src/critic.py - 自我评估 Critic 模块 (核心创新)"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from llama_index.llms.ollama import Ollama

from src.config import CriticConfig, LLMConfig


@dataclass
class CriticResult:
    """Critic 评估结果"""

    score: float  # 综合得分 1-5
    faithfulness_score: float  # 忠实度得分
    relevance_score: float  # 相关性得分
    feedback: str  # 诊断反馈
    needs_retry: bool  # 是否需要重试
    suggested_query: Optional[str] = None  # 建议的新查询


CRITIC_PROMPT_TEMPLATE = """你是一个严格的学术问答质量评估专家。请评估以下回答的质量。

## 用户问题
{question}

## 检索到的上下文
{context}

## 系统回答
{answer}

## 评估标准

1. **忠实度 (Faithfulness)**: 回答是否完全基于检索到的上下文？是否存在幻觉（编造不在上下文中的信息）？
   - 5分：完全基于上下文，无任何幻觉
   - 3分：大部分基于上下文，有少量推测
   - 1分：包含大量上下文中不存在的信息

2. **回答相关性 (Answer Relevance)**: 回答是否直接、准确地解答了用户的问题？
   - 5分：直接回答问题，信息完整
   - 3分：部分回答问题，信息不完整
   - 1分：答非所问或完全偏题

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：

{{
    "faithfulness_score": <1-5的数字>,
    "relevance_score": <1-5的数字>,
    "feedback": "<简短的诊断意见，说明存在的问题>",
    "suggested_query": "<如果分数低于4，建议一个更好的检索查询，否则为null>"
}}
"""


class Critic:
    """自我评估 Critic 模块

    核心创新：利用 LLM 作为裁判，自动评估 RAG 系统的回答质量，
    并在质量不达标时生成内部反馈和改进建议。
    """

    def __init__(self, critic_config: CriticConfig, llm_config: LLMConfig):
        self.config = critic_config
        self.llm = Ollama(
            model=llm_config.model_name,
            temperature=0.1,  # 评估需要稳定输出
            request_timeout=llm_config.request_timeout,
        )

    def evaluate(self, question: str, context: str, answer: str) -> CriticResult:
        """评估回答质量

        Args:
            question: 用户原始问题
            context: 检索到的上下文
            answer: 系统生成的回答

        Returns:
            CriticResult: 包含分数、反馈和是否需要重试的评估结果
        """
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=question, context=context, answer=answer
        )

        response = self.llm.complete(prompt)

        # 解析 LLM 输出
        result = self._parse_response(response.text)

        # 计算综合得分
        avg_score = (result["faithfulness_score"] + result["relevance_score"]) / 2

        return CriticResult(
            score=avg_score,
            faithfulness_score=result["faithfulness_score"],
            relevance_score=result["relevance_score"],
            feedback=result["feedback"],
            needs_retry=avg_score < self.config.threshold,
            suggested_query=result.get("suggested_query"),
        )

    def _parse_response(self, response_text: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        # 尝试提取 JSON
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 解析失败时返回默认值（保守策略：触发重试）
        return {
            "faithfulness_score": 3.0,
            "relevance_score": 3.0,
            "feedback": "无法解析评估结果，建议重新生成",
            "suggested_query": None,
        }
