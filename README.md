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







# CS6493 自然语言处理

# 小组项目进度报告

**项目名称：** 自动反思学术代理：一个用于学术论文问答的自我纠错对话系统

**选题：** Topic 3 - 使用 LlamaIndex 构建实用的 LLM 应用

**应用类型：** 对话代理 (Conversational Agent)

---

## 1. 引言与动机

传统的检索增强生成 (RAG) 系统在处理复杂的学术文档时往往表现不佳，由于检索不精确或上下文覆盖不完整，经常产生幻觉或无关的回答。本项目旨在开发一个**自动反思学术代理** —— 一个专门为学术论文问答设计的智能对话系统，具备自我评估和自我纠错能力。

### 1.1 问题陈述

当用户与基于 RAG 的学术论文问答系统交互时，通常会遇到以下问题：

- **幻觉**：系统编造源文档中不存在的信息
- **回答不完整**：回复只部分解答了用户的问题
- **上下文漂移**：在多轮对话中丢失相关上下文

### 1.2 解决方案与创新点

为了解决这些挑战并满足 "Advanced suggestions" 中关于实现基础人类反馈机制的要求，我们提出了一种新颖的**自我反思与自我纠错**机制。我们的系统不依赖手动用户反馈（这种方式效率低且劳动密集），而是采用自动化反馈循环：

1. **自动质量评估**：生成初始回答后，Critic 模块（由 LLM 驱动）基于 RAG 黄金三角指标自动评估回答质量
2. **自我纠错循环**：如果质量分数低于阈值，系统自主重写查询并重新生成回答
3. **迭代优化**：此过程持续进行，直到回答达到质量标准或达到最大重试次数

这种方法代表了 RLAIF（来自 AI 反馈的强化学习）或 Self-RAG 概念的早期实现，展示了无需人工干预的自主智能。

---

## 2. 系统架构与方法论

### 2.1 整体架构

系统基于 **LlamaIndex** 框架构建，由以下核心组件组成：

```
用户查询 → 检索器 → 草稿生成 → Critic 评估
                                    ↓
                          分数 >= 阈值?
                            ↓       ↓
                          是 → 返回答案
                            ↓
                          否 → 查询重写 → 重新检索与生成
                                      (最多 3 次迭代)
```

### 2.2 关键组件

#### 2.2.1 文档处理模块

- **PDF 读取**：使用 LlamaIndex 的 `SimpleDirectoryReader` 解析复杂的学术论文
- **分块策略**：
  - **固定大小分块**：512 tokens，10% 重叠（基线）
  - **语义分块**：按句子/段落边界感知分割（高级）

#### 2.2.2 向量索引构建器

- 嵌入模型：`BAAI/bge-small-zh-v1.5`，支持多语言
- 向量存储：LlamaIndex 内置的 `VectorStoreIndex`
- 支持索引持久化缓存

#### 2.2.3 LLM 后端（模型对比）

为满足对比至少两个 LLM 后端的要求：

- **主要模型**：Qwen-2.5-7B-Instruct（量化版，通过 Ollama 部署）
- **基线模型**：Qwen-2.5-1.5B-Instruct（轻量级对比）

两个模型都使用 **Ollama** 本地部署，满足 "Advanced suggestion" 中探索模型量化部署的要求。

#### 2.2.4 Critic 模块（核心创新）

Critic 模块是我们创新的核心。它基于以下指标评估回答：

- **忠实度分数 (1-5)**：回答是否基于检索到的上下文
- **回答相关性分数 (1-5)**：回答是否直接解答了问题

如果平均分数 < 4.0，Critic 会生成：

- 解释质量问题的诊断反馈
- 用于重新检索的建议优化查询

#### 2.2.5 对话记忆

- 使用 LlamaIndex 的 `ChatMemoryBuffer` 进行短期记忆管理
- 支持正确处理指代消解（例如："*这个方法*的缺点是什么？"）

### 2.3 自动反思工作流

```python
def chat(user_message):
    current_query = user_message
    for attempt in range(max_retries + 1):
        # 步骤 1: 检索相关文档块
        context = retriever.retrieve(current_query)
        
        # 步骤 2: 生成草稿答案
        answer = llm.generate(question=user_message, context=context)
        
        # 步骤 3: Critic 评估
        score, feedback, suggested_query = critic.evaluate(
            question=user_message,
            context=context,
            answer=answer
        )
        
        # 步骤 4: 检查是否达到质量阈值
        if score >= threshold:
            return answer
        
        # 步骤 5: 使用建议的查询进行下一次迭代
        if suggested_query:
            current_query = suggested_query
    
    return answer  # 达到最大重试次数后返回最佳结果
```

---

## 3. 评估计划

### 3.1 评估指标

我们专注于以 NLP 为中心的指标，而非硬件性能指标：

| 指标               | 描述                                             |
| ------------------ | ------------------------------------------------ |
| **上下文相关性**   | 检索到的文档块是否包含必要信息                   |
| **忠实度**         | 回答是否没有幻觉                                 |
| **回答相关性**     | 回答是否直接解答了问题                           |
| **任务完成率**     | 用户的需求是否最终得到满足                       |
| **自我纠错成功率** | 反思后分数提升的案例百分比（我们的核心创新指标） |

### 3.2 测试数据集

我们将构建一个包含 20-30 个学术问题的测试集，涵盖不同类型：

- 总结性问题（"主要贡献是什么？"）
- 方法论问题（"作者使用了什么方法？"）
- 事实性问题（"使用了哪些数据集？"）
- 比较性问题（"与基线相比如何？"）
- 多轮上下文问题（需要指代消解）

### 3.3 评估方法

- **自动评估**：使用 Critic 模块作为 LLM-as-a-judge
- **人工评估**：对部分回答进行人工抽查

---

## 4. 进度与已完成任务

### 4.1 已完成任务

| 任务           | 状态     | 描述                                            |
| -------------- | -------- | ----------------------------------------------- |
| 项目规划与设计 | ✅ 已完成 | 系统架构和方法论已确定                          |
| 环境搭建       | ✅ 已完成 | Python 虚拟环境及所有依赖                       |
| 配置模块       | ✅ 已完成 | `src/config.py` - 集中式配置管理                |
| 文档处理器     | ✅ 已完成 | `src/document_processor.py` - PDF 加载和分块    |
| 索引构建器     | ✅ 已完成 | `src/index_builder.py` - 向量索引构建           |
| Critic 模块    | ✅ 已完成 | `src/critic.py` - 自我评估机制（核心创新）      |
| 反思代理       | ✅ 已完成 | `src/reflective_agent.py` - 带自动纠错的主代理  |
| Streamlit UI   | ✅ 已完成 | `src/app.py` - 交互式演示界面                   |
| 评估框架       | ✅ 已完成 | `evaluation/evaluate.py` - RAG 黄金三角评估脚本 |
| 单元测试       | ✅ 已完成 | 15 个测试用例，全部通过                         |

### 4.2 代码统计

- **Python 文件总数**：10
- **代码总行数**：约 1,200 行
- **测试覆盖**：核心模块全面测试
- **Git 提交**：10 次结构化提交

### 4.3 当前项目结构

```
project/
├── src/
│   ├── config.py              # 配置管理
│   ├── document_processor.py  # PDF 处理与分块
│   ├── index_builder.py       # 向量索引构建
│   ├── critic.py              # 自我评估模块 ⭐
│   ├── reflective_agent.py    # 核心反思代理 ⭐
│   └── app.py                 # Streamlit 界面
├── evaluation/
│   ├── test_questions.json    # 测试问题集
│   └── evaluate.py            # 评估脚本
├── tests/                     # 单元测试（15 个用例）
├── docs/plans/                # 设计文档
└── README.md
```

---

## 5. 剩余工作与时间线

### 5.1 剩余任务

| 任务                          | 优先级 | 预计时间 |
| ----------------------------- | ------ | -------- |
| 使用真实学术论文进行实验      | 高     | 3 天     |
| 对比不同 LLM 后端的性能       | 高     | 2 天     |
| 对比分块策略（固定 vs. 语义） | 中     | 1 天     |
| 收集和分析评估指标            | 高     | 2 天     |
| 撰写包含实验结果的最终报告    | 高     | 3 天     |
| 准备演示幻灯片                | 中     | 1 天     |

### 5.2 时间线

- **第 1 周**：运行实验，收集数据
- **第 2 周**：分析结果，撰写最终报告
- **第 3 周**：准备演示，最终润色

---

## 6. 结论

我们已成功完成自动反思学术代理的核心实现。该系统具有创新的自我纠错机制，能够自动评估和改进回答质量，无需手动人类反馈。所有核心模块已实现并测试完毕，为实验评估阶段提供了坚实的基础。

关键创新 —— 基于 Critic 的自动反馈循环 —— 使我们的方法区别于传统的 RAG 系统，展示了 AI 驱动的质量保证在对话代理中的实际应用。

---

## 参考文献

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
2. Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv*.
3. LlamaIndex Documentation. https://docs.llamaindex.ai/
4. Ollama Documentation. https://ollama.ai/

