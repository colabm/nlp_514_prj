"""tests/test_document_processor.py"""
import pytest
from pathlib import Path


def test_document_processor_init():
    """测试 DocumentProcessor 初始化"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig

    config = ChunkingConfig()
    processor = DocumentProcessor(config)
    assert hasattr(processor, "load_pdf")
    assert hasattr(processor, "chunk_documents")
    assert hasattr(processor, "process")


def test_fixed_chunking_config():
    """测试固定分块策略配置"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig

    config = ChunkingConfig(strategy="fixed", chunk_size=256, chunk_overlap=25)
    processor = DocumentProcessor(config)
    assert processor.config.chunk_size == 256
    assert processor.config.chunk_overlap == 25
    assert processor.config.strategy == "fixed"


def test_semantic_chunking_config():
    """测试语义分块策略配置"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig

    config = ChunkingConfig(strategy="semantic")
    processor = DocumentProcessor(config)
    assert processor.config.strategy == "semantic"


def test_load_pdf_file_not_found():
    """测试加载不存在的 PDF 抛出错误"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig

    processor = DocumentProcessor(ChunkingConfig())
    with pytest.raises(FileNotFoundError):
        processor.load_pdf("nonexistent.pdf")
