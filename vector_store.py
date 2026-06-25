"""
vector_store.py
----------------
Builds a FAISS vector store from local documents using HuggingFace
sentence-transformer embeddings. This powers the agent's "local knowledge"
tool, separate from its web search tool.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store() -> FAISS:
    """
    Loads all .txt files from /data, splits them into overlapping chunks,
    embeds them, and builds (or loads a cached) FAISS index.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(INDEX_DIR):
        print("[vector_store] Loading cached FAISS index...")
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    print("[vector_store] Building new FAISS index from /data ...")
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_DIR, filename))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[vector_store] Split into {len(chunks)} chunks.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    print("[vector_store] Index built and cached.")
    return vector_store


def get_retriever(k: int = 3):
    """Returns a retriever that fetches the top-k most relevant chunks."""
    store = build_vector_store()
    return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Quick manual test: python vector_store.py
    retriever = get_retriever()
    results = retriever.invoke("What is the equipment allowance for remote work?")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200])
