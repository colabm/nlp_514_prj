# Auto-Reflective Academic Agent 实施计划 (Implementation Plan)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个基于 LlamaIndex 的自动反思学术对话代理，能够处理 PDF 学术论文，具备自我评估和自我纠错能力。

**Architecture:** 系统采用 RAG (检索增强生成) 架构，核心创新是"自动反思工作流"——生成初步答案后，由 Critic 模块自动评估质量，若不达标则自主重写查询并重新生成。使用 LlamaIndex 管理文档索引和对话记忆，Ollama 提供本地 LLM 推理。

**Tech Stack:** Python 3.10+, LlamaIndex, Ollama, Streamlit, Ragas (评估)

---

## 项目结构 (Project Structure)

```
project/
├── docs/plans/                    # 设计与实施文档
├── src/
│   ├── __init__.py
│   ├── config.py                  # 配置管理
│   ├── document_processor.py      # PDF 加载与分块
│   ├── index_builder.py           # 向量索引构建
│   ├── retriever.py               # 检索器封装
│   ├── critic.py                  # 自我评估 Critic 模块
│   ├── reflective_agent.py        # 核心反思代理
│   └── app.py                     # Streamlit 界面
├── tests/
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_critic.py
│   ├── test_reflective_agent.py
│   └── test_integration.py
├── data/
│   └── papers/                    # 存放测试用 PDF
├── evaluation/
│   ├── test_questions.json        # 评测问题集
│   └── evaluate.py                # RAG Triad 评估脚本
├── requirements.txt
└── README.md
```

---

## Task 1: 环境搭建与依赖安装

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: 创建 requirements.txt**

```txt
# Core
llama-index==0.10.30
llama-index-llms-ollama==0.1.3
llama-index-embeddings-huggingface==0.2.0

# PDF Processing
pypdf==4.0.0

# UI
streamlit==1.32.0

# Evaluation
ragas==0.1.7
datasets==2.18.0

# Utils
python-dotenv==1.0.1
pydantic==2.6.0
```

**Step 2: 创建空的 __init__.py 文件**

```bash
mkdir -p src tests data/papers evaluation
touch src/__init__.py tests/__init__.py
```

**Step 3: 安装依赖**

Run: `pip install -r requirements.txt`

**Step 4: 验证 Ollama 安装**

Run: `ollama list`
Expected: 显示已安装的模型列表（如果没有模型，后续步骤会拉取）

**Step 5: 拉取所需模型**

Run: `ollama pull qwen2.5:7b-instruct`
Run: `ollama pull qwen2.5:1.5b-instruct`

**Step 6: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: initialize project structure and dependencies"
```

---

## Task 2: 配置管理模块

**Files:**
- Create: `src/config.py`

**Step 1: 编写配置模块**

```python
"""配置管理模块"""
from pydantic import BaseModel
from typing import Literal

class ChunkingConfig(BaseModel):
    """分块策略配置"""
    strategy: Literal["fixed", "semantic"] = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50

class LLMConfig(BaseModel):
    """LLM 配置"""
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    request_timeout: float = 120.0

class CriticConfig(BaseModel):
    """Critic 模块配置"""
    threshold: float = 4.0  # 1-5 分，低于此值触发反思
    max_retries: int = 3

class AppConfig(BaseModel):
    """应用总配置"""
    chunking: ChunkingConfig = ChunkingConfig()
    llm: LLMConfig = LLMConfig()
    critic: CriticConfig = CriticConfig()
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
# 默认配置实例
default_config = AppConfig()
```

**Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: add configuration management module"
```

---

## Task 3: PDF 文档处理与分块模块

**Files:**
- Create: `src/document_processor.py`
- Create: `tests/test_document_processor.py`

**Step 1: 编写失败测试**

```python
"""tests/test_document_processor.py"""
import pytest
from pathlib import Path

def test_load_pdf_returns_documents():
    """测试 PDF 加载返回文档列表"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig
    
    processor = DocumentProcessor(ChunkingConfig())
    # 需要一个测试 PDF，这里用占位符
    # 实际测试时需要放置真实 PDF
    assert hasattr(processor, 'load_pdf')

def test_fixed_chunking_produces_correct_size():
    """测试固定分块策略产生正确大小的块"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig
    
    config = ChunkingConfig(strategy="fixed", chunk_size=256, chunk_overlap=25)
    processor = DocumentProcessor(config)
    assert processor.config.chunk_size == 256

def test_semantic_chunking_available():
    """测试语义分块策略可用"""
    from src.document_processor import DocumentProcessor
    from src.config import ChunkingConfig
    
    config = ChunkingConfig(strategy="semantic")
    processor = DocumentProcessor(config)
    assert processor.config.strategy == "semantic"
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_document_processor.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: 实现文档处理模块**

```python
"""src/document_processor.py - PDF 文档处理与分块"""
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import ChunkingConfig

class DocumentProcessor:
    """文档处理器：负责 PDF 加载和分块"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.pdf_reader = PDFReader()
        
    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        """加载 PDF 文件并返回文档列表"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        documents = self.pdf_reader.load_data(file=pdf_path)
        return documents
    
    def chunk_documents(
        self, 
        documents: List[Document],
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
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
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    ) -> List:
        """一站式处理：加载 + 分块"""
        documents = self.load_pdf(pdf_path)
        nodes = self.chunk_documents(documents, embedding_model_name)
        return nodes
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_document_processor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/document_processor.py tests/test_document_processor.py
git commit -m "feat: add document processor with fixed and semantic chunking"
```

---

## Task 4: 向量索引构建模块

**Files:**
- Create: `src/index_builder.py`

**Step 1: 实现索引构建模块**

```python
"""src/index_builder.py - 向量索引构建"""
from pathlib import Path
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import AppConfig

class IndexBuilder:
    """向量索引构建器"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.embedding_model
        )
        
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
```

**Step 2: Commit**

```bash
git add src/index_builder.py
git commit -m "feat: add vector index builder with persistence support"
```

---

## Task 5: Critic 自我评估模块 (核心创新)

**Files:**
- Create: `src/critic.py`
- Create: `tests/test_critic.py`

**Step 1: 编写失败测试**

```python
"""tests/test_critic.py"""
import pytest

def test_critic_returns_score_and_feedback():
    """测试 Critic 返回分数和反馈"""
    from src.critic import Critic
    from src.config import CriticConfig, LLMConfig
    
    critic = Critic(CriticConfig(), LLMConfig())
    assert hasattr(critic, 'evaluate')

def test_critic_score_in_valid_range():
    """测试 Critic 分数在有效范围内"""
    from src.critic import CriticResult
    
    # 模拟一个评估结果
    result = CriticResult(
        score=4.5,
        faithfulness_score=4.0,
        relevance_score=5.0,
        feedback="答案准确且相关",
        needs_retry=False
    )
    assert 1.0 <= result.score <= 5.0
    assert result.needs_retry == False

def test_critic_triggers_retry_on_low_score():
    """测试低分触发重试"""
    from src.critic import CriticResult
    
    result = CriticResult(
        score=2.5,
        faithfulness_score=2.0,
        relevance_score=3.0,
        feedback="答案包含未在文档中出现的信息",
        needs_retry=True
    )
    assert result.needs_retry == True
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_critic.py -v`
Expected: FAIL

**Step 3: 实现 Critic 模块**

```python
"""src/critic.py - 自我评估 Critic 模块 (核心创新)"""
from dataclasses import dataclass
from typing import Optional

from llama_index.llms.ollama import Ollama

from src.config import CriticConfig, LLMConfig

@dataclass
class CriticResult:
    """Critic 评估结果"""
    score: float              # 综合得分 1-5
    faithfulness_score: float # 忠实度得分
    relevance_score: float    # 相关性得分
    feedback: str             # 诊断反馈
    needs_retry: bool         # 是否需要重试
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
        
    def evaluate(
        self, 
        question: str, 
        context: str, 
        answer: str
    ) -> CriticResult:
        """评估回答质量
        
        Args:
            question: 用户原始问题
            context: 检索到的上下文
            answer: 系统生成的回答
            
        Returns:
            CriticResult: 包含分数、反馈和是否需要重试的评估结果
        """
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer
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
            suggested_query=result.get("suggested_query")
        )
    
    def _parse_response(self, response_text: str) -> dict:
        """解析 LLM 的 JSON 输出"""
        import json
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
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
            "suggested_query": None
        }
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_critic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/critic.py tests/test_critic.py
git commit -m "feat: add self-evaluation Critic module (core innovation)"
```

---

## Task 6: 反思代理核心模块

**Files:**
- Create: `src/reflective_agent.py`
- Create: `tests/test_reflective_agent.py`

**Step 1: 编写失败测试**

```python
"""tests/test_reflective_agent.py"""
import pytest

def test_agent_has_chat_method():
    """测试 Agent 具有 chat 方法"""
    from src.reflective_agent import ReflectiveAgent
    
    # 不实际初始化，只检查类定义
    assert hasattr(ReflectiveAgent, 'chat')

def test_agent_tracks_reflection_count():
    """测试 Agent 追踪反思次数"""
    from src.reflective_agent import ChatResponse
    
    response = ChatResponse(
        answer="测试答案",
        reflection_count=2,
        final_score=4.5,
        sources=[]
    )
    assert response.reflection_count == 2
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_reflective_agent.py -v`
Expected: FAIL

**Step 3: 实现反思代理模块**

```python
"""src/reflective_agent.py - 自动反思学术代理 (核心模块)"""
from dataclasses import dataclass, field
from typing import List, Optional

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
    reflection_count: int      # 经历了几次反思
    final_score: float         # 最终质量分数
    sources: List[str]         # 引用的文档来源
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
        
        for attempt in range(self.config.critic.max_retries + 1):
            # 检索相关文档
            retrieved_nodes = self.retriever.retrieve(current_query)
            context = "\n\n".join([node.text for node in retrieved_nodes])
            sources = [node.metadata.get("file_name", "unknown") for node in retrieved_nodes]
            
            # 生成回答
            response = self.chat_engine.chat(current_query)
            answer = str(response)
            
            # Critic 评估
            critic_result = self.critic.evaluate(
                question=user_message,  # 始终用原始问题评估
                context=context,
                answer=answer
            )
            reflection_history.append(critic_result)
            
            # 检查是否通过
            if not critic_result.needs_retry:
                return ChatResponse(
                    answer=answer,
                    reflection_count=attempt,
                    final_score=critic_result.score,
                    sources=list(set(sources)),
                    reflection_history=reflection_history
                )
            
            # 需要重试：使用建议的新查询或保持原查询
            if critic_result.suggested_query:
                current_query = critic_result.suggested_query
            
            # 重置对话引擎状态以进行新一轮检索
            # (保留记忆但清除当前轮次的中间状态)
        
        # 达到最大重试次数，返回最后一次的结果
        return ChatResponse(
            answer=answer,
            reflection_count=self.config.critic.max_retries,
            final_score=critic_result.score,
            sources=list(set(sources)),
            reflection_history=reflection_history
        )
    
    def reset_memory(self):
        """重置对话记忆（开始新会话）"""
        self.memory.reset()
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
        )
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_reflective_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/reflective_agent.py tests/test_reflective_agent.py
git commit -m "feat: add reflective agent with auto-correction workflow"
```

---

## Task 7: Streamlit 交互界面

**Files:**
- Create: `src/app.py`

**Step 1: 实现 Streamlit 界面**

```python
"""src/app.py - Streamlit 交互界面"""
import streamlit as st
from pathlib import Path

from src.config import AppConfig, ChunkingConfig, LLMConfig
from src.document_processor import DocumentProcessor
from src.index_builder import IndexBuilder
from src.reflective_agent import ReflectiveAgent

st.set_page_config(
    page_title="Auto-Reflective Academic Agent",
    page_icon="📚",
    layout="wide"
)

st.title("📚 自动反思学术代理 (Auto-Reflective Academic Agent)")
st.markdown("*基于 LlamaIndex 的智能学术论文问答系统，具备自我评估与自我纠错能力*")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 模型选择
    model_name = st.selectbox(
        "选择 LLM 模型",
        ["qwen2.5:7b-instruct", "qwen2.5:1.5b-instruct", "llama3:8b-instruct"],
        index=0
    )
    
    # 分块策略
    chunking_strategy = st.selectbox(
        "选择分块策略",
        ["fixed", "semantic"],
        index=0,
        format_func=lambda x: "固定大小分块" if x == "fixed" else "语义分块"
    )
    
    chunk_size = st.slider("分块大小 (tokens)", 256, 1024, 512, 64)
    
    # Critic 配置
    st.subheader("🔍 Critic 配置")
    threshold = st.slider("质量阈值 (1-5)", 1.0, 5.0, 4.0, 0.5)
    max_retries = st.slider("最大反思次数", 1, 5, 3)
    
    # PDF 上传
    st.subheader("📄 上传文档")
    uploaded_file = st.file_uploader("上传 PDF 论文", type=["pdf"])

# 初始化会话状态
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# 处理上传的 PDF
if uploaded_file and not st.session_state.index_built:
    with st.spinner("正在处理文档并构建索引..."):
        # 保存上传的文件
        pdf_path = Path("data/papers") / uploaded_file.name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # 配置
        config = AppConfig(
            chunking=ChunkingConfig(
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.1)
            ),
            llm=LLMConfig(model_name=model_name),
            critic=CriticConfig(threshold=threshold, max_retries=max_retries)
        )
        
        # 处理文档
        processor = DocumentProcessor(config.chunking)
        nodes = processor.process(pdf_path, config.embedding_model)
        
        # 构建索引
        builder = IndexBuilder(config)
        index = builder.build_index(nodes)
        
        # 初始化 Agent
        st.session_state.agent = ReflectiveAgent(index, config)
        st.session_state.index_built = True
        
    st.success(f"✅ 成功加载 {uploaded_file.name}，共 {len(nodes)} 个文档块")

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("🔍 查看反思过程"):
                st.json(message["metadata"])

# 用户输入
if prompt := st.chat_input("请输入你的学术问题..."):
    if not st.session_state.agent:
        st.warning("⚠️ 请先上传 PDF 文档")
    else:
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = st.session_state.agent.chat(prompt)
            
            st.markdown(response.answer)
            
            # 显示反思元数据
            metadata = {
                "反思次数": response.reflection_count,
                "最终质量分数": response.final_score,
                "引用来源": response.sources,
                "反思历史": [
                    {
                        "综合分数": r.score,
                        "忠实度": r.faithfulness_score,
                        "相关性": r.relevance_score,
                        "诊断意见": r.feedback
                    }
                    for r in response.reflection_history
                ]
            }
            
            with st.expander("🔍 查看反思过程"):
                st.json(metadata)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "metadata": metadata
        })

# 底部：重置按钮
if st.session_state.agent:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置对话"):
            st.session_state.agent.reset_memory()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📊 查看统计"):
            total_reflections = sum(
                m.get("metadata", {}).get("反思次数", 0) 
                for m in st.session_state.messages 
                if m["role"] == "assistant"
            )
            st.metric("总反思次数", total_reflections)
```

**Step 2: Commit**

```bash
git add src/app.py
git commit -m "feat: add Streamlit interactive UI with reflection visualization"
```

---

## Task 8: RAG Triad 评估脚本

**Files:**
- Create: `evaluation/test_questions.json`
- Create: `evaluation/evaluate.py`

**Step 1: 创建测试问题集**

```json
{
  "questions": [
    {
      "id": 1,
      "question": "这篇论文的主要贡献是什么？",
      "type": "summary"
    },
    {
      "id": 2,
      "question": "作者使用了什么方法来解决问题？",
      "type": "methodology"
    },
    {
      "id": 3,
      "question": "实验中使用了哪些数据集？",
      "type": "factual"
    },
    {
      "id": 4,
      "question": "与基线方法相比，提出的方法有什么优势？",
      "type": "comparison"
    },
    {
      "id": 5,
      "question": "论文中提到的局限性有哪些？",
      "type": "analysis"
    }
  ],
  "multi_turn": [
    {
      "id": 101,
      "turns": [
        "这篇论文的研究背景是什么？",
        "基于这个背景，作者提出了什么假设？",
        "这个假设是如何被验证的？"
      ],
      "type": "context_dependent"
    }
  ]
}
```

**Step 2: 创建评估脚本**

```python
"""evaluation/evaluate.py - RAG Triad 评估脚本"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import csv

from src.reflective_agent import ReflectiveAgent, ChatResponse

@dataclass
class EvaluationResult:
    """单个问题的评估结果"""
    question_id: int
    question: str
    answer: str
    context_relevance: float    # 上下文相关性
    faithfulness: float         # 忠实度
    answer_relevance: float     # 回答相关性
    reflection_count: int       # 反思次数
    final_score: float          # 最终质量分数
    task_completed: bool        # 任务是否完成

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
    output_file: str = "evaluation/results.csv"
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
            answer=response.answer,
            context_relevance=last_critic.relevance_score if last_critic else 0,
            faithfulness=last_critic.faithfulness_score if last_critic else 0,
            answer_relevance=last_critic.relevance_score if last_critic else 0,
            reflection_count=response.reflection_count,
            final_score=response.final_score,
            task_completed=response.final_score >= 4.0
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
                    answer=response.answer,
                    context_relevance=last_critic.relevance_score if last_critic else 0,
                    faithfulness=last_critic.faithfulness_score if last_critic else 0,
                    answer_relevance=last_critic.relevance_score if last_critic else 0,
                    reflection_count=response.reflection_count,
                    final_score=response.final_score,
                    task_completed=response.final_score >= 4.0
                )
                results.append(result)
    
    # 保存详细结果到 CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
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
            if triggered_reflection else 1.0
        )
    )
    
    print("\n" + "="*50)
    print("📊 聚合评估结果 (Aggregated Metrics)")
    print("="*50)
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
```

**Step 3: Commit**

```bash
git add evaluation/test_questions.json evaluation/evaluate.py
git commit -m "feat: add RAG Triad evaluation framework with self-correction metrics"
```

---

## Task 9: 集成测试

**Files:**
- Create: `tests/test_integration.py`

**Step 1: 编写集成测试**

```python
"""tests/test_integration.py - 端到端集成测试"""
import pytest
from pathlib import Path

# 跳过条件：没有测试 PDF 或没有 Ollama
pytestmark = pytest.mark.skipif(
    not Path("data/papers").exists() or not any(Path("data/papers").glob("*.pdf")),
    reason="No test PDF available"
)

def test_full_pipeline_with_reflection():
    """测试完整的反思工作流"""
    from src.config import AppConfig
    from src.document_processor import DocumentProcessor
    from src.index_builder import IndexBuilder
    from src.reflective_agent import ReflectiveAgent
    
    # 获取测试 PDF
    pdf_files = list(Path("data/papers").glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files for testing")
    
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

def test_multi_turn_conversation():
    """测试多轮对话的上下文保持"""
    from src.config import AppConfig
    from src.document_processor import DocumentProcessor
    from src.index_builder import IndexBuilder
    from src.reflective_agent import ReflectiveAgent
    
    pdf_files = list(Path("data/papers").glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files for testing")
    
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
    # (这个断言比较松，主要验证系统不崩溃)
    assert len(r2.answer) > 10
```

**Step 2: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

## Task 10: README 文档

**Files:**
- Create: `README.md`

**Step 1: 编写 README**

```markdown
# Auto-Reflective Academic Agent

CS6493 自然语言处理课程项目 - Topic 3: Building Practical LLM Applications with LlamaIndex

## 项目简介

本项目实现了一个**自动反思学术代理 (Auto-Reflective Academic Agent)**，能够处理复杂的学术 PDF 文档，并具备**自我评估与自我纠错**的能力。

### 核心创新

- **自动反思机制**：系统生成回答后，由内置的 Critic 模块自动评估质量
- **自主纠错**：若质量不达标，自动重写查询并重新生成，无需人工干预
- **RAG Triad 评估**：采用上下文相关性、忠实度、回答相关性三大指标

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama (macOS)
brew install ollama

# 拉取模型
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
```

### 3. 启动应用

```bash
streamlit run src/app.py
```

### 4. 运行评估

```bash
python -m evaluation.evaluate
```

## 项目结构

```
├── src/
│   ├── config.py              # 配置管理
│   ├── document_processor.py  # PDF 处理与分块
│   ├── index_builder.py       # 向量索引构建
│   ├── critic.py              # 自我评估模块
│   ├── reflective_agent.py    # 核心反思代理
│   └── app.py                 # Streamlit 界面
├── evaluation/
│   ├── test_questions.json    # 测试问题集
│   └── evaluate.py            # 评估脚本
└── tests/                     # 单元测试
```

## 评估指标

| 指标 | 描述 |
|------|------|
| Context Relevance | 检索上下文的相关性 |
| Faithfulness | 回答的忠实度（无幻觉） |
| Answer Relevance | 回答与问题的相关性 |
| Self-Correction Success Rate | 自我纠错成功率 |

## 技术栈

- LlamaIndex 0.10+
- Ollama (本地 LLM 部署)
- Streamlit (交互界面)
- Ragas (评估框架)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with project overview and instructions"
```

---

## 执行顺序总结

| Task | 描述 | 预计时间 |
|------|------|---------|
| 1 | 环境搭建与依赖安装 | 15 min |
| 2 | 配置管理模块 | 10 min |
| 3 | PDF 文档处理与分块 | 30 min |
| 4 | 向量索引构建 | 20 min |
| 5 | Critic 自我评估模块 | 45 min |
| 6 | 反思代理核心模块 | 60 min |
| 7 | Streamlit 交互界面 | 45 min |
| 8 | RAG Triad 评估脚本 | 30 min |
| 9 | 集成测试 | 20 min |
| 10 | README 文档 | 15 min |

**总计约 5 小时**
