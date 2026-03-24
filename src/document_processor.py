"""src/document_processor.py - PDF 文档处理与分块"""
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import ChunkingConfig


class DocumentProcessor:
    """文档处理器：负责 PDF 加载和分块"""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        """加载 PDF 文件并返回文档列表"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 使用 SimpleDirectoryReader 加载单个文件
        reader = SimpleDirectoryReader(input_files=[str(pdf_path)])
        documents = reader.load_data()
        return documents

    def chunk_documents(
        self,
        documents: List[Document],
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> List:
        """根据配置的策略对文档进行分块"""
        if self.config.strategy == "fixed":
            # 固定大小分块
            splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        elif self.config.strategy == "semantic":
            # 语义分块 - 需要 embedding 模型
            embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
            splitter = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

        nodes = splitter.get_nodes_from_documents(documents)
        return nodes

    def process(
        self,
        pdf_path: str | Path,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> List:
        """一站式处理：加载 + 分块"""
        documents = self.load_pdf(pdf_path)
        nodes = self.chunk_documents(documents, embedding_model_name)
        return nodes
