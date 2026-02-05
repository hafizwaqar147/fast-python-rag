# pip install langchain langchain-community langchain-openai langchain-huggingface faiss-cpu sentence-transformers pypdf

import os
import sys
import argparse
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document

# --- CONFIGURATION ---
DATA_DIR = "./data"
INDEX_PATH = "./faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Set via environment or change here for local OpenAI-compatible API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def load_and_index():
    """Load documents from /data, split, and cache in FAISS."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return None

    docs: List[Document] = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    if not docs:
        print("No .pdf or .txt files found in /data.")
        return None

    # Fast recursive splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # Embeddings (cached locally by LangFace/FAISS logic)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create or Load Index
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(INDEX_PATH)
    
    return vectorstore

def get_rag_chain(vectorstore):
    """Setup minimal RAG chain."""
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        openai_api_key=OPENAI_API_KEY, 
        openai_api_base=OPENAI_BASE_URL,
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Use the following pieces of context to answer the question at the end.
If the answer is not in the context, say: "Information not available in provided documents."
Do not attempt to answer using external knowledge.

Context: {context}

Question: {question}
Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

def main():
    parser = argparse.ArgumentParser(description="Fast Minimal RAG")
    parser.add_argument("query", type=str, help="The question to ask the RAG system")
    args = parser.parse_args()

    # Low latency check: Load index
    vectorstore = load_and_index()
    if not vectorstore:
        return

    # Run query
    chain = get_rag_chain(vectorstore)
    response = chain.invoke(args.query)
    
    # Print clean output
    print(f"\nQUERY: {args.query}")
    print(f"ANSWER: {response.content}\n")

if __name__ == "__main__":
    main()
