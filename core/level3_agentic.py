"""
Level 3: Agentic RAG Pipeline
-------------------------------
Core logic extracted from terminal_scripts/AgenticRAG.py.
- LLM Agentic Router (GREETING vs RESEARCH intent)
- LLM-powered Query Expansion
- Two-stage retrieval: Bi-Encoder (wide net) → Cross-Encoder (precision re-rank)
"""

import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# INGESTION
# ==========================================
def ingest(pdf_path, bi_encoder):
    """
    Read a PDF, apply recursive splitting, embed, and store in ChromaDB.
    Returns (client, collection, num_chunks).
    """
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    my_chunks = []
    for page in doc:
        my_chunks.extend(text_splitter.split_text(page.get_text()))
    doc.close()

    embeddings = bi_encoder.encode(my_chunks).tolist()

    client = chromadb.PersistentClient(path="./scholar_db_level3")
    try:
        client.delete_collection(name="scholar_rag_v3")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="scholar_rag_v3")
    chunk_ids = [f"id{i}" for i in range(len(my_chunks))]
    collection.add(documents=my_chunks, embeddings=embeddings, ids=chunk_ids)

    return client, collection, len(my_chunks)


def load_existing_db():
    """Load an existing Level 3 DB from disk."""
    try:
        client = chromadb.PersistentClient(path="./scholar_db_level3")
        collection = client.get_collection(name="scholar_rag_v3")
        if collection.count() > 0:
            return client, collection
    except Exception:
        pass
    return None


# ==========================================
# AGENTIC ROUTER
# ==========================================
def route_query(user_query, llm):
    """Classify user intent as GREETING or RESEARCH."""
    routing_prompt = f"""
    You are a smart routing agent for a document retrieval system.
    Analyze the user's input and classify it into exactly one of these two categories:

    1. GREETING: The user is saying hello, asking how you are, making small talk, or saying goodbye.
    2. RESEARCH: The user is asking a specific question, requesting a summary, or asking for facts.

    User input: "{user_query}"

    Output ONLY the category name (GREETING or RESEARCH) and absolutely nothing else.
    """
    response = llm.generate_content(routing_prompt, generation_config={"temperature": 0.0})
    return response.text.strip().upper()


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
def query(user_question, collection, bi_encoder, cross_encoder, llm):
    """
    Run the Agentic RAG pipeline.
    Returns a dict with the answer and all intermediate debug data.
    """
    # 1. Route intent
    route = route_query(user_question, llm)

    if "GREETING" in route:
        return {
            "answer": "Hello! I am ScholarRAG, an autonomous research assistant. Ask me anything about the loaded document!",
            "route": "GREETING",
            "expanded_queries": [],
            "bi_encoder_chunks": [],
            "cross_encoder_scores": [],
            "final_chunks": [],
        }

    # 2. Expand the query
    search_queries = expand_query(user_question, llm)
    query_embeddings = bi_encoder.encode(search_queries).tolist()

    # 3. Bi-Encoder wide retrieval
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=5
    )

    unique_chunks = set()
    for document_list in results['documents']:
        for document in document_list:
            unique_chunks.add(document)
    retrieved_chunks = list(unique_chunks)

    # 4. Cross-Encoder re-ranking
    sentence_pairs = [[user_question, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(sentence_pairs)

    scored_chunks = list(zip([float(s) for s in scores], retrieved_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    best_chunks = [chunk for score, chunk in scored_chunks[:4]]

    # 5. Synthesize
    context_string = "\n\n".join(best_chunks)
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
        "route": "RESEARCH",
        "expanded_queries": search_queries,
        "bi_encoder_chunks": retrieved_chunks,
        "cross_encoder_scores": scored_chunks,
        "final_chunks": best_chunks,
    }
