"""src/index_builder.py - 向量索引构建"""
from pathlib import Path
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import AppConfig


class IndexBuilder:
    """向量索引构建器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)

    def build_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """从文档节点构建向量索引"""
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embed_model,
            show_progress=True,
        )
        return index

    def save_index(self, index: VectorStoreIndex, persist_dir: str | Path):
        """持久化索引到本地"""
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))

    def load_index(self, persist_dir: str | Path) -> Optional[VectorStoreIndex]:
        """从本地加载已有索引"""
        persist_dir = Path(persist_dir)
        if not persist_dir.exists():
            return None

        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(
            storage_context,
            embed_model=self.embed_model,
        )
        return index
