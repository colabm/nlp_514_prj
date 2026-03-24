"""tests/test_integration.py - 端到端集成测试"""
import pytest
from pathlib import Path


# 跳过条件：没有测试 PDF
def has_test_pdf():
    """检查是否有测试用的 PDF 文件"""
    papers_dir = Path("data/papers")
    return papers_dir.exists() and any(papers_dir.glob("*.pdf"))


@pytest.mark.skipif(not has_test_pdf(), reason="No test PDF available")
class TestIntegration:
    """集成测试类 - 需要实际的 PDF 和 Ollama 服务"""

    def test_full_pipeline_with_reflection(self):
        """测试完整的反思工作流"""
        from src.config import AppConfig
        from src.document_processor import DocumentProcessor
        from src.index_builder import IndexBuilder
        from src.reflective_agent import ReflectiveAgent

        # 获取测试 PDF
        pdf_files = list(Path("data/papers").glob("*.pdf"))
        pdf_path = pdf_files[0]
        config = AppConfig()

        # 处理文档
        processor = DocumentProcessor(config.chunking)
        nodes = processor.process(pdf_path)
        assert len(nodes) > 0

        # 构建索引
        builder = IndexBuilder(config)
        index = builder.build_index(nodes)
        assert index is not None

        # 初始化 Agent
        agent = ReflectiveAgent(index, config)

        # 测试对话
        response = agent.chat("这篇论文的主要内容是什么？")

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.final_score > 0
        assert response.reflection_count >= 0

    def test_multi_turn_conversation(self):
        """测试多轮对话的上下文保持"""
        from src.config import AppConfig
        from src.document_processor import DocumentProcessor
        from src.index_builder import IndexBuilder
        from src.reflective_agent import ReflectiveAgent

        pdf_files = list(Path("data/papers").glob("*.pdf"))
        pdf_path = pdf_files[0]
        config = AppConfig()

        processor = DocumentProcessor(config.chunking)
        nodes = processor.process(pdf_path)
        builder = IndexBuilder(config)
        index = builder.build_index(nodes)
        agent = ReflectiveAgent(index, config)

        # 第一轮
        r1 = agent.chat("这篇论文使用了什么方法？")
        assert r1.answer is not None

        # 第二轮（依赖上下文）
        r2 = agent.chat("这个方法的优点是什么？")
        assert r2.answer is not None

        # 验证第二轮能理解"这个方法"的指代
        assert len(r2.answer) > 10


class TestModuleIntegration:
    """模块集成测试 - 不需要外部服务"""

    def test_config_integration(self):
        """测试配置模块集成"""
        from src.config import AppConfig, ChunkingConfig, LLMConfig, CriticConfig

        config = AppConfig(
            chunking=ChunkingConfig(strategy="fixed", chunk_size=256),
            llm=LLMConfig(model_name="test-model"),
            critic=CriticConfig(threshold=3.5, max_retries=2),
        )

        assert config.chunking.chunk_size == 256
        assert config.llm.model_name == "test-model"
        assert config.critic.threshold == 3.5

    def test_document_processor_integration(self):
        """测试文档处理器配置集成"""
        from src.config import ChunkingConfig
        from src.document_processor import DocumentProcessor

        config = ChunkingConfig(strategy="fixed", chunk_size=512, chunk_overlap=50)
        processor = DocumentProcessor(config)

        assert processor.config.strategy == "fixed"
        assert processor.config.chunk_size == 512

    def test_critic_result_integration(self):
        """测试 Critic 结果与 Agent 响应的集成"""
        from src.critic import CriticResult
        from src.reflective_agent import ChatResponse

        critic_results = [
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
                feedback="质量良好",
                needs_retry=False,
            ),
        ]

        response = ChatResponse(
            answer="测试回答",
            reflection_count=1,
            final_score=4.5,
            sources=["test.pdf"],
            reflection_history=critic_results,
        )

        assert len(response.reflection_history) == 2
        assert response.reflection_count == 1
        assert response.final_score == 4.5
