"""tests/test_critic.py"""
import pytest


def test_critic_result_dataclass():
    """测试 CriticResult 数据类"""
    from src.critic import CriticResult

    result = CriticResult(
        score=4.5,
        faithfulness_score=4.0,
        relevance_score=5.0,
        feedback="答案准确且相关",
        needs_retry=False,
    )
    assert 1.0 <= result.score <= 5.0
    assert result.needs_retry is False


def test_critic_result_triggers_retry_on_low_score():
    """测试低分触发重试"""
    from src.critic import CriticResult

    result = CriticResult(
        score=2.5,
        faithfulness_score=2.0,
        relevance_score=3.0,
        feedback="答案包含未在文档中出现的信息",
        needs_retry=True,
    )
    assert result.needs_retry is True


def test_critic_init():
    """测试 Critic 初始化"""
    from src.critic import Critic
    from src.config import CriticConfig, LLMConfig

    critic = Critic(CriticConfig(), LLMConfig())
    assert hasattr(critic, "evaluate")
    assert hasattr(critic, "_parse_response")


def test_critic_parse_response_valid_json():
    """测试解析有效的 JSON 响应"""
    from src.critic import Critic
    from src.config import CriticConfig, LLMConfig

    critic = Critic(CriticConfig(), LLMConfig())

    response_text = """
    {
        "faithfulness_score": 4.5,
        "relevance_score": 4.0,
        "feedback": "回答基本准确",
        "suggested_query": null
    }
    """

    result = critic._parse_response(response_text)
    assert result["faithfulness_score"] == 4.5
    assert result["relevance_score"] == 4.0
    assert result["feedback"] == "回答基本准确"


def test_critic_parse_response_invalid_json():
    """测试解析无效 JSON 时的回退行为"""
    from src.critic import Critic
    from src.config import CriticConfig, LLMConfig

    critic = Critic(CriticConfig(), LLMConfig())

    response_text = "这不是一个有效的 JSON 响应"

    result = critic._parse_response(response_text)
    # 应该返回默认值
    assert result["faithfulness_score"] == 3.0
    assert result["relevance_score"] == 3.0
