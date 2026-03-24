# CS6493 课程项目设计文档 (Design Document)

**项目选题**: Topic 3 - Building Practical LLM Applications with LlamaIndex
**应用类型**: Conversational Agent (对话代理)
**项目愿景**: Auto-Reflective Academic Agent (自动反思学术代理)

## 1. 项目背景与创新点 (Introduction & Novelty)

传统的检索增强生成 (RAG) 对话系统在处理长篇复杂的学术 PDF 时，常常因为检索片段不准确或未能完全覆盖问题而产生幻觉 (Hallucination) 或无关回答。本项目旨在开发一个面向学术领域的“自动反思对话代理 (Auto-Reflective Academic Agent)”。

**核心创新 (Novelty for High Score)**：
为了满足 "Advanced suggestions" 中关于 "basic human feedback mechanisms" 的要求，并提升系统的自主性 (Autonomous Intelligence)，本项目**不采用依赖人工点击的低效反馈方式**，而是引入了**“自我反思与自我纠错 (Self-Reflection and Self-Correction)”** 机制 (RLAIF/Self-RAG 雏形)。
当系统生成初步回答后，会利用 LLM 扮演“裁判 (Critic)”，依据 RAG 黄金三角指标对答案进行内部评估。若得分低于阈值，系统会生成内部反馈 (Internal Feedback)，自主重写查询 (Query Rewriting) 并重新检索生成，直至输出高质量的最终答案。

## 2. 系统架构设计 (Methodology & System Architecture)

本系统基于 **LlamaIndex** 框架构建，其核心数据流如下：

*   **数据接入 (Data Ingestion)**: 使用 LlamaIndex 的 `PDFReader` 解析复杂的学术论文。
*   **文档处理与分块 (Document Processing & Chunking)**:
    *   **策略 A**: 基础的固定大小分块 (Fixed-Size Chunking, e.g., 512 tokens + 10% overlap)。
    *   **策略 B**: 高级的语义分块 (Semantic Chunking, 按句子或段落边界分割)，以对比不同分块策略对下游检索质量的影响。
*   **模型后端 (LLM Backends) & 量化部署 (Quantization)**:
    *   利用 **Ollama** 进行本地部署，满足计算资源约束的 "Advanced suggestions"。
    *   **对比模型组**:
        1.  **强推理模型**: 例如 `Qwen-2.5-7B-Instruct` 或 `Llama-3-8B-Instruct` (4-bit/8-bit 量化版)，负责生成和反思。
        2.  **轻量级基线模型**: 例如小型化的 Qwen1.5-1.8B 或特定量化格式的变体，用于对比在较弱推理能力下“自我反思”机制的效能差异。
*   **对话记忆管理 (Conversational Memory)**:
    *   集成 LlamaIndex 的 `ChatMemoryBuffer`。确保多轮学术问答中（如指代消解：“那个方法的缺点是什么？”）的上下文能够被准确继承和重构。

## 3. 核心机制：自动反思工作流 (Auto-Reflective Workflow)

1.  **初次检索与生成 (Draft Generation)**: 根据用户查询 Q 检索文档块 C，生成草稿答案 A_draft。
2.  **自我裁判 (Self-Evaluation)**: Critic 模块（LLM）根据 Q、C 和 A_draft，评估**忠实度 (Faithfulness)** 和 **回答相关性 (Answer Relevance)**，输出分数 S (1-5分)。
3.  **反馈循环 (Feedback Loop)**:
    *   若 S >= 4 (阈值): 将 A_draft 作为最终答案返回给用户。
    *   若 S < 4: Critic 模块生成内部诊断意见 (例如：“答案包含了未在文档中提及的外部知识”或“未回答问题的第二部分”)。系统根据该反馈重新生成 Q_new，执行新一轮的检索与生成，直至达标或达到最大重试次数 (Max Retries = 3)。

## 4. 能力评估与指标体系 (Capability Evaluation & Experiments)

为了高度契合“自然语言处理 (NLP)”的学术标准，本项目摒弃了内存、TTFT 等底层硬件指标，将评估重心完全聚焦于**文本质量、检索效果以及反思机制的有效性**。

我们将构建一个包含 20-30 个复杂学术提问的测试集，并采用以下指标进行评估 (可通过 LLM-as-a-judge 工具如 Ragas/TruLens 或高质量开源裁判模型自动评分，辅以人工抽查)：

*   **RAG 黄金三角 (RAG Triad - Text Quality)**:
    1.  **Context Relevance (上下文相关性)**: 检索到的文档块是否包含了回答该问题所需的全部必要信息。
    2.  **Faithfulness (忠实度/无幻觉率)**: 最终答案是否 100% 根植于检索到的上下文，没有捏造事实。
    3.  **Answer Relevance (回答相关性)**: 最终答案是否直接、准确、无冗余地解答了用户的原始提问。
*   **多轮对话指标 (Multi-turn Contextual Metrics)**:
    *   **Task Completion Rate (任务完成率)**: 对话最终是否满足了用户（哪怕经过了多次追问）的真实需求。
*   **【核心亮点】反思机制效能指标 (Reflection Mechanism Metrics)**:
    *   **Self-Correction Success Rate (自我纠错成功率)**: 系统触发内部反思的频率，以及**经过反思纠错后，RAG 黄金三角得分的平均提升幅度 (Delta Score)**。这将是证明该创新点有效性的最强力证据。

## 5. 项目产出 (Deliverables)

1.  包含完整系统架构与反思机制实现的 Python 源码 (Jupyter Notebook / .py)。
2.  基于 Streamlit 或 Gradio 的轻量级交互界面，用于演示。
3.  详尽的 Progress Report (5页内) 与 Final Report (6页正文内，包含实验图表和上述评估指标的深度讨论)。
4.  用于课堂展示的 15 分钟 Presentation Slides。