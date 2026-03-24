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
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama (macOS)
brew install ollama

# 启动 Ollama 服务
ollama serve

# 拉取模型 (在另一个终端)
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:1.5b-instruct
```

### 3. 启动应用

```bash
source venv/bin/activate
streamlit run src/app.py
```

### 4. 运行测试

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### 5. 运行评估

```bash
source venv/bin/activate
python -c "
from src.config import AppConfig
from src.document_processor import DocumentProcessor
from src.index_builder import IndexBuilder
from src.reflective_agent import ReflectiveAgent
from evaluation.evaluate import evaluate_agent

# 加载你的 PDF
config = AppConfig()
processor = DocumentProcessor(config.chunking)
nodes = processor.process('data/papers/your_paper.pdf')

builder = IndexBuilder(config)
index = builder.build_index(nodes)

agent = ReflectiveAgent(index, config)
metrics = evaluate_agent(agent)
"
```

## 项目结构

```
├── src/
│   ├── config.py              # 配置管理
│   ├── document_processor.py  # PDF 处理与分块
│   ├── index_builder.py       # 向量索引构建
│   ├── critic.py              # 自我评估模块 (核心创新)
│   ├── reflective_agent.py    # 核心反思代理
│   └── app.py                 # Streamlit 界面
├── evaluation/
│   ├── test_questions.json    # 测试问题集
│   └── evaluate.py            # 评估脚本
├── tests/                     # 单元测试
├── data/papers/               # PDF 文档存放目录
└── docs/plans/                # 设计文档
```

## 系统架构

```
用户问题 → 检索器 → 生成初步回答 → Critic 评估
                                      ↓
                              分数 >= 阈值?
                              ↓         ↓
                            是 → 返回答案
                              ↓
                            否 → 重写查询 → 重新检索生成 (最多3次)
```

## 评估指标

| 指标 | 描述 |
|------|------|
| Context Relevance | 检索上下文的相关性 |
| Faithfulness | 回答的忠实度（无幻觉） |
| Answer Relevance | 回答与问题的相关性 |
| Self-Correction Success Rate | 自我纠错成功率 (核心创新指标) |

## 技术栈

- **LlamaIndex** - RAG 框架
- **Ollama** - 本地 LLM 部署
- **Streamlit** - 交互界面
- **HuggingFace Embeddings** - 文本向量化

## 配置说明

可在 `src/config.py` 中调整以下参数：

- `chunk_size`: 文档分块大小 (默认 512)
- `chunk_overlap`: 分块重叠 (默认 50)
- `strategy`: 分块策略 ("fixed" 或 "semantic")
- `threshold`: Critic 质量阈值 (默认 4.0)
- `max_retries`: 最大反思次数 (默认 3)
