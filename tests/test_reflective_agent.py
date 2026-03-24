"""tests/test_reflective_agent.py"""
import pytest


def test_chat_response_dataclass():
    """测试 ChatResponse 数据类"""
    from src.reflective_agent import ChatResponse

    response = ChatResponse(
        answer="测试答案",
        reflection_count=2,
        final_score=4.5,
        sources=["test.pdf"],
    )
    assert response.answer == "测试答案"
    assert response.reflection_count == 2
    assert response.final_score == 4.5
    assert "test.pdf" in response.sources


def test_chat_response_with_reflection_history():
    """测试带反思历史的 ChatResponse"""
    from src.reflective_agent import ChatResponse
    from src.critic import CriticResult

    history = [
        CriticResult(
            score=3.0,
            faithfulness_score=3.0,
            relevance_score=3.0,
            feedback="需要改进",
            needs_retry=True,
        ),
        CriticResult(
            score=4.5,
            faithfulness_score=4.5,
            relevance_score=4.5,
            feedback="很好",
            needs_retry=False,
        ),
    ]

    response = ChatResponse(
        answer="改进后的答案",
        reflection_count=1,
        final_score=4.5,
        sources=[],
        reflection_history=history,
    )
    assert len(response.reflection_history) == 2
    assert response.reflection_history[0].needs_retry is True
    assert response.reflection_history[1].needs_retry is False


def test_reflective_agent_has_required_methods():
    """测试 ReflectiveAgent 具有必需的方法"""
    from src.reflective_agent import ReflectiveAgent

    assert hasattr(ReflectiveAgent, "chat")
    assert hasattr(ReflectiveAgent, "reset_memory")
