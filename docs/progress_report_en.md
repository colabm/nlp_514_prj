# CS6493 Natural Language Processing
# Group Project Progress Report

**Project Title:** Auto-Reflective Academic Agent: A Self-Correcting Conversational System for Academic Paper Q&A

**Topic:** Topic 3 - Building Practical LLM Applications with LlamaIndex

**Application Type:** Conversational Agent

---

## 1. Introduction and Motivation

Traditional Retrieval-Augmented Generation (RAG) systems often struggle with complex academic documents, frequently producing hallucinations or irrelevant responses due to imprecise retrieval or incomplete context coverage. This project aims to develop an **Auto-Reflective Academic Agent** — an intelligent conversational system specifically designed for academic paper question-answering that incorporates self-evaluation and self-correction capabilities.

### 1.1 Problem Statement

When users interact with RAG-based Q&A systems on academic papers, they commonly encounter:
- **Hallucinations**: The system fabricates information not present in the source documents
- **Incomplete answers**: Responses that only partially address the user's question
- **Context drift**: Loss of relevant context in multi-turn conversations

### 1.2 Proposed Solution and Innovation

To address these challenges and fulfill the "Advanced suggestions" requirement for implementing basic human feedback mechanisms, we propose a novel **Self-Reflection and Self-Correction** mechanism. Instead of relying on manual user feedback (which is inefficient and labor-intensive), our system employs an automated feedback loop:

1. **Automated Quality Assessment**: After generating an initial response, a Critic module (powered by an LLM) automatically evaluates the answer quality based on the RAG Triad metrics
2. **Self-Correction Loop**: If the quality score falls below a threshold, the system autonomously rewrites the query and regenerates the response
3. **Iterative Refinement**: This process continues until the answer meets quality standards or reaches the maximum retry limit

This approach represents an early-stage implementation of RLAIF (Reinforcement Learning from AI Feedback) or Self-RAG concepts, demonstrating autonomous intelligence without human intervention.

---

## 2. System Architecture and Methodology

### 2.1 Overall Architecture

The system is built on the **LlamaIndex** framework and consists of the following core components:

```
User Query → Retriever → Draft Generation → Critic Evaluation
                                                    ↓
                                          Score >= Threshold?
                                            ↓           ↓
                                          Yes → Return Answer
                                            ↓
                                          No → Query Rewriting → Re-retrieve & Regenerate
                                                              (max 3 iterations)
```

### 2.2 Key Components

#### 2.2.1 Document Processing Module
- **PDF Ingestion**: Using LlamaIndex's `SimpleDirectoryReader` to parse complex academic papers
- **Chunking Strategies**: 
  - **Fixed-size chunking**: 512 tokens with 10% overlap (baseline)
  - **Semantic chunking**: Sentence/paragraph boundary-aware splitting (advanced)

#### 2.2.2 Vector Index Builder
- Embedding model: `BAAI/bge-small-zh-v1.5` for multilingual support
- Vector store: LlamaIndex's built-in `VectorStoreIndex`
- Persistence support for index caching

#### 2.2.3 LLM Backends (Model Comparison)
To fulfill the requirement of comparing at least two LLM backends:
- **Primary Model**: Qwen-2.5-7B-Instruct (quantized, via Ollama)
- **Baseline Model**: Qwen-2.5-1.5B-Instruct (lightweight comparison)

Both models are deployed locally using **Ollama**, satisfying the "Advanced suggestion" for exploring model quantization deployment.

#### 2.2.4 Critic Module (Core Innovation)
The Critic module is the heart of our innovation. It evaluates responses based on:
- **Faithfulness Score (1-5)**: Whether the answer is grounded in the retrieved context
- **Answer Relevance Score (1-5)**: Whether the answer directly addresses the question

If the average score < 4.0, the Critic generates:
- Diagnostic feedback explaining the quality issues
- A suggested refined query for re-retrieval

#### 2.2.5 Conversational Memory
- Using LlamaIndex's `ChatMemoryBuffer` for short-term memory management
- Enables proper handling of coreference resolution (e.g., "What are the drawbacks of *this method*?")

### 2.3 Auto-Reflective Workflow

```python
def chat(user_message):
    current_query = user_message
    for attempt in range(max_retries + 1):
        # Step 1: Retrieve relevant document chunks
        context = retriever.retrieve(current_query)
        
        # Step 2: Generate draft answer
        answer = llm.generate(question=user_message, context=context)
        
        # Step 3: Critic evaluation
        score, feedback, suggested_query = critic.evaluate(
            question=user_message,
            context=context,
            answer=answer
        )
        
        # Step 4: Check if quality threshold is met
        if score >= threshold:
            return answer
        
        # Step 5: Use suggested query for next iteration
        if suggested_query:
            current_query = suggested_query
    
    return answer  # Return best effort after max retries
```

---

## 3. Evaluation Plan

### 3.1 Evaluation Metrics

We focus on NLP-centric metrics rather than hardware performance metrics:

| Metric | Description |
|--------|-------------|
| **Context Relevance** | Whether retrieved chunks contain necessary information |
| **Faithfulness** | Whether the answer is free from hallucinations |
| **Answer Relevance** | Whether the answer directly addresses the question |
| **Task Completion Rate** | Whether the user's need is ultimately satisfied |
| **Self-Correction Success Rate** | Percentage of cases where reflection improves the score (our core innovation metric) |

### 3.2 Test Dataset

We will construct a test set of 20-30 academic questions across different types:
- Summary questions ("What is the main contribution?")
- Methodology questions ("What method did the authors use?")
- Factual questions ("Which datasets were used?")
- Comparative questions ("How does this compare to baselines?")
- Multi-turn contextual questions (requiring coreference resolution)

### 3.3 Evaluation Method

- **Automated evaluation**: LLM-as-a-judge using the Critic module
- **Human evaluation**: Manual spot-checking of a subset of responses

---

## 4. Progress and Completed Tasks

### 4.1 Completed Tasks

| Task | Status | Description |
|------|--------|-------------|
| Project planning and design | ✅ Completed | System architecture and methodology finalized |
| Environment setup | ✅ Completed | Python virtual environment with all dependencies |
| Configuration module | ✅ Completed | `src/config.py` - Centralized configuration management |
| Document processor | ✅ Completed | `src/document_processor.py` - PDF loading and chunking |
| Index builder | ✅ Completed | `src/index_builder.py` - Vector index construction |
| Critic module | ✅ Completed | `src/critic.py` - Self-evaluation mechanism (core innovation) |
| Reflective agent | ✅ Completed | `src/reflective_agent.py` - Main agent with auto-correction |
| Streamlit UI | ✅ Completed | `src/app.py` - Interactive demonstration interface |
| Evaluation framework | ✅ Completed | `evaluation/evaluate.py` - RAG Triad evaluation script |
| Unit tests | ✅ Completed | 15 test cases, all passing |

### 4.2 Code Statistics

- **Total Python files**: 10
- **Total lines of code**: ~1,200
- **Test coverage**: Core modules fully tested
- **Git commits**: 10 structured commits

### 4.3 Current Project Structure

```
project/
├── src/
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # PDF processing & chunking
│   ├── index_builder.py       # Vector index construction
│   ├── critic.py              # Self-evaluation module ⭐
│   ├── reflective_agent.py    # Core reflective agent ⭐
│   └── app.py                 # Streamlit interface
├── evaluation/
│   ├── test_questions.json    # Test question set
│   └── evaluate.py            # Evaluation script
├── tests/                     # Unit tests (15 cases)
├── docs/plans/                # Design documents
└── README.md
```

---

## 5. Remaining Work and Timeline

### 5.1 Remaining Tasks

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Conduct experiments with real academic papers | High | 3 days |
| Compare performance across different LLM backends | High | 2 days |
| Compare chunking strategies (fixed vs. semantic) | Medium | 1 day |
| Collect and analyze evaluation metrics | High | 2 days |
| Write final report with experimental results | High | 3 days |
| Prepare presentation slides | Medium | 1 day |

### 5.2 Timeline

- **Week 1**: Run experiments, collect data
- **Week 2**: Analyze results, write final report
- **Week 3**: Prepare presentation, final polishing

---

## 6. Conclusion

We have successfully completed the core implementation of our Auto-Reflective Academic Agent. The system features an innovative self-correction mechanism that automatically evaluates and improves response quality without requiring manual human feedback. All core modules have been implemented and tested, providing a solid foundation for the experimental evaluation phase.

The key innovation — the automated Critic-based feedback loop — distinguishes our approach from traditional RAG systems and demonstrates a practical implementation of AI-driven quality assurance in conversational agents.

---

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
2. Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv*.
3. LlamaIndex Documentation. https://docs.llamaindex.ai/
4. Ollama Documentation. https://ollama.ai/

