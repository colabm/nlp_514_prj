"""src/critic.py - 自我评估 Critic 模块 (核心创新)"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from llama_index.llms.ollama import Ollama

from config import CriticConfig, LLMConfig


@dataclass
class CriticResult:
    """Critic 评估结果"""

    score: float  # 综合得分 1-5
    faithfulness_score: float  # 忠实度得分
    relevance_score: float  # 相关性得分
    feedback: str  # 诊断反馈
    needs_retry: bool  # 是否需要重试
    suggested_query: Optional[str] = None  # 建议的新查询


CRITIC_PROMPT_TEMPLATE = """你是一个极其严格的学术问答质量审核员。你的职责是找出回答中的所有问题。
宁可误判为低质量，也不能放过任何有问题的回答。5分应该非常罕见，只有完美的回答才能获得5分。

## 用户问题
{question}

## 检索到的上下文
{context}

## 系统回答
{answer}

## 审核步骤

请严格按以下步骤进行审核：

### 第一步：幻觉检查 (Hallucination Check)
逐句检查系统回答，找出所有在上下文中**找不到依据**的陈述。
- 列出每一个可疑的幻觉陈述
- 如果回答中包含上下文未提及的具体数字、名称、日期等，这很可能是幻觉

### 第二步：完整性检查 (Completeness Check)  
分析用户问题，检查回答是否**完整解答**了问题的所有部分。
- 用户问了什么？
- 哪些部分被回答了？
- 哪些部分被遗漏了？

### 第三步：准确性检查 (Accuracy Check)
检查回答是否**曲解或误解**了上下文中的信息。
- 有没有张冠李戴？
- 有没有过度概括或错误推断？

## 评分标准（必须严格遵守）

### 忠实度评分 (Faithfulness Score)
- **5分**：完美 - 每一句话都能在上下文中找到明确依据，零幻觉（极其罕见）
- **4分**：良好 - 基本忠实，最多有1处轻微的不确定表述
- **3分**：一般 - 有1-2处明显幻觉或无依据的陈述
- **2分**：较差 - 有3处以上幻觉，或有严重的事实错误
- **1分**：很差 - 大部分内容是编造的，严重不可信

### 回答相关性评分 (Relevance Score)
- **5分**：完美 - 直接、完整、准确地回答了问题的所有方面（极其罕见）
- **4分**：良好 - 回答了主要问题，但有1处小遗漏或不够深入
- **3分**：一般 - 只回答了问题的部分内容，有明显遗漏
- **2分**：较差 - 回答偏离主题，或大部分问题未被解答
- **1分**：很差 - 完全答非所问，或回答无意义

## 输出格式

请严格按以下 JSON 格式输出，analysis 字段必须详细列出发现的问题：

{{
    "analysis": {{
        "hallucinations": ["<列出发现的每一个幻觉或无依据陈述，如果没有则为空数组>"],
        "missing_parts": ["<列出用户问题中未被回答的部分，如果没有则为空数组>"],
        "inaccuracies": ["<列出曲解或误解上下文的地方，如果没有则为空数组>"]
    }},
    "faithfulness_score": <1-5的整数>,
    "relevance_score": <1-5的整数>,
    "feedback": "<基于上述分析，总结主要问题>",
    "suggested_query": "<如果分数低于4，建议一个更好的检索查询来获取缺失的信息，否则为null>"
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
                parsed = json.loads(json_match.group())
                
                # 如果有 analysis 字段，基于分析结果增强 feedback
                if "analysis" in parsed:
                    analysis = parsed["analysis"]
                    issues = []
                    
                    hallucinations = analysis.get("hallucinations", [])
                    if hallucinations:
                        issues.append(f"幻觉问题({len(hallucinations)}处): {'; '.join(hallucinations[:2])}")
                    
                    missing = analysis.get("missing_parts", [])
                    if missing:
                        issues.append(f"遗漏内容: {'; '.join(missing[:2])}")
                    
                    inaccuracies = analysis.get("inaccuracies", [])
                    if inaccuracies:
                        issues.append(f"准确性问题: {'; '.join(inaccuracies[:2])}")
                    
                    # 如果分析发现了问题但分数过高，自动降低分数
                    total_issues = len(hallucinations) + len(missing) + len(inaccuracies)
                    if total_issues > 0:
                        # 有问题时，确保分数不会过高
                        max_faith = max(1, 5 - len(hallucinations) - len(inaccuracies))
                        max_relev = max(1, 5 - len(missing))
                        parsed["faithfulness_score"] = min(parsed.get("faithfulness_score", 5), max_faith)
                        parsed["relevance_score"] = min(parsed.get("relevance_score", 5), max_relev)
                        
                        if issues and not parsed.get("feedback"):
                            parsed["feedback"] = " | ".join(issues)
                
                return parsed
            except json.JSONDecodeError:
                pass

        # 解析失败时返回默认值（保守策略：触发重试）
        return {
            "faithfulness_score": 2.0,
            "relevance_score": 2.0,
            "feedback": "无法解析评估结果，建议重新生成",
            "suggested_query": None,
        }
