"""
Level 2: Intermediate RAG Pipeline
------------------------------------
Core logic extracted from terminal_scripts/IntermediateRAG.py.
- Recursive semantic chunking (preserves paragraph boundaries)
- Page-level metadata tagging
- LLM-powered Query Expansion (1 original + 3 variations)
- Optional metadata filtering by page number
"""

import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# INGESTION
# ==========================================
def ingest(pdf_path, bi_encoder):
    """
    Read a PDF page-by-page, apply recursive splitting,
    tag metadata, embed, and store in ChromaDB.
    Returns (client, collection, num_chunks, total_pages).
    """
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    my_chunks = []
    my_metadata = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_chunks = text_splitter.split_text(page_text)
        my_chunks.extend(page_chunks)
        for _ in range(len(page_chunks)):
            my_metadata.append({"page": page_num + 1})

    total_pages = len(doc)
    doc.close()

    embeddings = bi_encoder.encode(my_chunks).tolist()

    client = chromadb.PersistentClient(path="./scholar_db_intermediate")
    try:
        client.delete_collection(name="scholar_rag_v2")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="scholar_rag_v2")
    chunk_ids = [f"id{i}" for i in range(len(my_chunks))]
    collection.add(
        documents=my_chunks,
        embeddings=embeddings,
        metadatas=my_metadata,
        ids=chunk_ids
    )

    return client, collection, len(my_chunks), total_pages


def load_existing_db():
    """Load an existing Intermediate DB from disk."""
    try:
        client = chromadb.PersistentClient(path="./scholar_db_intermediate")
        collection = client.get_collection(name="scholar_rag_v2")
        if collection.count() > 0:
            return client, collection
    except Exception:
        pass
    return None


# ==========================================
# QUERY EXPANSION
# ==========================================
def expand_query(original_query, llm):
    """Use the LLM to generate 3 alternative search queries."""
    expansion_prompt = f"""
    You are an expert AI research assistant. The user is searching a vector database for a technical research paper.
    Their original query is: "{original_query}"

    Generate 3 highly targeted, short search queries to find this information.
    - Use specific technical keywords or synonyms.
    - Keep each query UNDER 6 words.
    - Do NOT write full sentences or generic academic phrases.

    Output exactly 3 alternative queries, one per line. No bullet points or intro text.
    """
    response = llm.generate_content(expansion_prompt, generation_config={"temperature": 0.2})
    expanded_queries = [q.strip() for q in response.text.split('\n') if q.strip()]
    return [original_query] + expanded_queries


# ==========================================
# QUERY
# ==========================================
def query(user_question, collection, bi_encoder, llm, page_filter=None):
    """
    Run the Intermediate RAG pipeline with query expansion and optional page filtering.
    Returns a dict with the answer and all intermediate debug data.
    """
    # 1. Expand query
    search_queries = expand_query(user_question, llm)
    query_embeddings = bi_encoder.encode(search_queries).tolist()

    # 2. Build search params
    search_params = {
        "query_embeddings": query_embeddings,
        "n_results": 2
    }
    if page_filter is not None:
        search_params["where"] = {"page": int(page_filter)}

    # 3. Execute search
    results = collection.query(**search_params)

    # 4. Flatten and deduplicate
    unique_chunks = set()
    for document_list in results['documents']:
        for document in document_list:
            unique_chunks.add(document)
    retrieved_chunks = list(unique_chunks)
    context_string = "\n\n".join(retrieved_chunks)

    # 5. Synthesize
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
        "expanded_queries": search_queries,
        "retrieved_chunks": retrieved_chunks,
        "page_filter_applied": page_filter,
    }
