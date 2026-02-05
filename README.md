# Fast Python RAG System 🚀

A high-performance, production-ready Retrieval-Augmented Generation (RAG) system implemented in a single Python file. This system is designed for low latency and minimal complexity, prioritizing speed and efficiency.

## 🌟 Key Features

- **Fast Recursive Chunking**: Optimizes document splitting for maximum performance.
- **Local In-Memory Vector Store**: Uses FAISS for lightning-fast semantic search without external dependencies.
- **Local Embeddings**: Cached locally using `sentence-transformers/all-MiniLM-L6-v2` for privacy and speed.
- **OpenAI-Compatible Chat Model**: Seamlessly integrates with any OpenAI-compatible API (LLM).
- **Automated Caching**: FAISS index is saved locally to avoid redundant document processing.
- **Grounded Answers**: Strictly answers based on provided context or states information is not available.

## 🛠️ Components

| Component | Responsibility |
| :--- | :--- |
| `PyPDFLoader` / `TextLoader` | Loads documents from the `./data` directory. |
| `RecursiveCharacterTextSplitter` | Breaks documents into 500-character chunks with a 50-character overlap. |
| `HuggingFaceEmbeddings` | Generates semantic vectors using `all-MiniLM-L6-v2`. |
| `FAISS` | Handles vector indexing, local storage, and similarity search. |
| `ChatOpenAI` | Interfaces with the LLM to generate responses. |
| `PromptTemplate` | Ensures answers are grounded and context-aware. |

## 📋 Prerequisites

- Python 3.10+
- OpenAI API Key (or local equivalent)
- `OPENAI_API_KEY` environment variable set.
- `OPENAI_BASE_URL` environment variable set (optional, defaults to OpenAI).

## 🚀 Installation

Install the required dependencies using pip:

```bash
pip install langchain langchain-community langchain-openai langchain-huggingface faiss-cpu sentence-transformers pypdf
```

## 📖 Usage

1. **Prepare Data**: Place your `.pdf` and `.txt` files in the `./data` directory.
2. **Run the System**: Execute the script with your question as an argument.

```bash
python rag.py "How do I set up the RAG system?"
```

## 💡 Important Features Required (Future Enhancements)

- **Dynamic Data Refresh**: Automatically re-index when new files are added to `./data`.
- **Hybrid Search**: Combine semantic search with keyword search (BM25) for better precision.
- **API Wrapper**: Wrap the system in a Fast API or Flask endpoint for production use.
- **Evaluation Framework**: Implement RAGAS or similar to measure retrieval quality.
- **Persistent Cache**: Use a more robust database strategy if document size scales beyond 1GB.

---
*Created as part of the Fast Python RAG project.*
