"""evaluation/evaluate.py - RAG Triad 评估脚本"""
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from src.reflective_agent import ReflectiveAgent


@dataclass
class EvaluationResult:
    """单个问题的评估结果"""

    question_id: int
    question: str
    answer: str
    context_relevance: float  # 上下文相关性
    faithfulness: float  # 忠实度
    answer_relevance: float  # 回答相关性
    reflection_count: int  # 反思次数
    final_score: float  # 最终质量分数
    task_completed: bool  # 任务是否完成


@dataclass
class AggregatedMetrics:
    """聚合指标"""

    avg_context_relevance: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_reflection_count: float
    task_completion_rate: float
    self_correction_success_rate: float  # 核心创新指标


def evaluate_agent(
    agent: ReflectiveAgent,
    questions_file: str = "evaluation/test_questions.json",
    output_file: str = "evaluation/results.csv",
) -> AggregatedMetrics:
    """对 Agent 进行全面评估"""

    # 加载测试问题
    with open(questions_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results: List[EvaluationResult] = []

    # 评估单轮问题
    for q in test_data["questions"]:
        agent.reset_memory()  # 每个问题独立评估

        response = agent.chat(q["question"])

        # 使用最后一次 Critic 评估的分数
        last_critic = response.reflection_history[-1] if response.reflection_history else None

        result = EvaluationResult(
            question_id=q["id"],
            question=q["question"],
            answer=response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
            context_relevance=last_critic.relevance_score if last_critic else 0,
            faithfulness=last_critic.faithfulness_score if last_critic else 0,
            answer_relevance=last_critic.relevance_score if last_critic else 0,
            reflection_count=response.reflection_count,
            final_score=response.final_score,
            task_completed=response.final_score >= 4.0,
        )
        results.append(result)

        print(f"[{q['id']}] Score: {response.final_score:.2f}, Reflections: {response.reflection_count}")

    # 评估多轮对话
    for mq in test_data.get("multi_turn", []):
        agent.reset_memory()

        for i, turn in enumerate(mq["turns"]):
            response = agent.chat(turn)

            if i == len(mq["turns"]) - 1:  # 只记录最后一轮
                last_critic = response.reflection_history[-1] if response.reflection_history else None

                result = EvaluationResult(
                    question_id=mq["id"],
                    question=" -> ".join(mq["turns"]),
                    answer=response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
                    context_relevance=last_critic.relevance_score if last_critic else 0,
                    faithfulness=last_critic.faithfulness_score if last_critic else 0,
                    answer_relevance=last_critic.relevance_score if last_critic else 0,
                    reflection_count=response.reflection_count,
                    final_score=response.final_score,
                    task_completed=response.final_score >= 4.0,
                )
                results.append(result)

    # 保存详细结果到 CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # 计算聚合指标
    n = len(results)

    # 计算自我纠错成功率：触发反思且最终通过的比例
    triggered_reflection = [r for r in results if r.reflection_count > 0]
    successful_corrections = [r for r in triggered_reflection if r.task_completed]

    metrics = AggregatedMetrics(
        avg_context_relevance=sum(r.context_relevance for r in results) / n,
        avg_faithfulness=sum(r.faithfulness for r in results) / n,
        avg_answer_relevance=sum(r.answer_relevance for r in results) / n,
        avg_reflection_count=sum(r.reflection_count for r in results) / n,
        task_completion_rate=sum(1 for r in results if r.task_completed) / n,
        self_correction_success_rate=(
            len(successful_corrections) / len(triggered_reflection)
            if triggered_reflection
            else 1.0
        ),
    )

    print("\n" + "=" * 50)
    print("📊 聚合评估结果 (Aggregated Metrics)")
    print("=" * 50)
    print(f"平均上下文相关性: {metrics.avg_context_relevance:.2f}")
    print(f"平均忠实度: {metrics.avg_faithfulness:.2f}")
    print(f"平均回答相关性: {metrics.avg_answer_relevance:.2f}")
    print(f"平均反思次数: {metrics.avg_reflection_count:.2f}")
    print(f"任务完成率: {metrics.task_completion_rate:.1%}")
    print(f"🌟 自我纠错成功率: {metrics.self_correction_success_rate:.1%}")

    return metrics


if __name__ == "__main__":
    # 示例用法（需要先构建 Agent）
    print("请在主程序中调用 evaluate_agent(agent) 进行评估")
    print("\n示例用法:")
    print("  from evaluation.evaluate import evaluate_agent")
    print("  metrics = evaluate_agent(agent)")
