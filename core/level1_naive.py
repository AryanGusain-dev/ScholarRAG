"""
Level 1: Naive RAG Pipeline
----------------------------
Core logic extracted from terminal_scripts/NaiveRAG.py.
- Fixed character chunking (1000 chars, 100 overlap)
- Bi-Encoder similarity search (Top 3)
- Strict anti-hallucination prompt
"""

import fitz
import chromadb


# ==========================================
# CHUNKING (Naive fixed-size)
# ==========================================
def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


# ==========================================
# INGESTION
# ==========================================
def ingest(pdf_path, bi_encoder):
    """
    Read a PDF, split into naive chunks, embed, and store in ChromaDB.
    Returns (chroma_client, collection, num_chunks).
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = chunk_text(full_text)
    embeddings = bi_encoder.encode(chunks).tolist()

    chroma_client = chromadb.PersistentClient(path="./scholar_db_naive")
    # Clear old data if re-ingesting
    try:
        chroma_client.delete_collection(name="scholar_rag_naive")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="scholar_rag_naive")
    chunk_ids = [f"id{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=chunk_ids)

    return chroma_client, collection, len(chunks)


def load_existing_db():
    """Load an existing Naive DB from disk. Returns (client, collection) or None."""
    try:
        client = chromadb.PersistentClient(path="./scholar_db_naive")
        collection = client.get_collection(name="scholar_rag_naive")
        if collection.count() > 0:
            return client, collection
    except Exception:
        pass
    return None


# ==========================================
# QUERY
# ==========================================
def query(user_question, collection, bi_encoder, llm):
    """
    Run the Naive RAG pipeline.
    Returns a dict with the answer and all intermediate debug data.
    """
    question_embedding = bi_encoder.encode([user_question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results['documents'][0]
    context_string = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an expert research assistant. Your job is to answer the user's question using ONLY the provided context from a research paper.

    If the answer is not contained in the context below, you must strictly reply with: "I'm sorry, but I cannot answer this based on the provided documents." Do not use outside knowledge.

    Context:
    {context_string}

    User Question:
    {user_question}
    """

    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "retrieved_chunks": retrieved_chunks,
    }
