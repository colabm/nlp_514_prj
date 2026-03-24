"""src/app.py - Streamlit 交互界面"""
import streamlit as st
from pathlib import Path

from src.config import AppConfig, ChunkingConfig, LLMConfig, CriticConfig
from src.document_processor import DocumentProcessor
from src.index_builder import IndexBuilder
from src.reflective_agent import ReflectiveAgent

st.set_page_config(
    page_title="Auto-Reflective Academic Agent",
    page_icon="📚",
    layout="wide",
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
        index=0,
    )

    # 分块策略
    chunking_strategy = st.selectbox(
        "选择分块策略",
        ["fixed", "semantic"],
        index=0,
        format_func=lambda x: "固定大小分块" if x == "fixed" else "语义分块",
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
                chunk_overlap=int(chunk_size * 0.1),
            ),
            llm=LLMConfig(model_name=model_name),
            critic=CriticConfig(threshold=threshold, max_retries=max_retries),
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
                        "诊断意见": r.feedback,
                    }
                    for r in response.reflection_history
                ],
            }

            with st.expander("🔍 查看反思过程"):
                st.json(metadata)

        st.session_state.messages.append(
            {"role": "assistant", "content": response.answer, "metadata": metadata}
        )

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
