"""
Level 5: Self-Healing RAG Pipeline
----------------------------------
- Agentic Router (GREETING vs RESEARCH)
- Two-stage retrieval: Bi-Encoder → Cross-Encoder
- Self-Healing Gate: An LLM 'Grader' evaluates if the database context contains the answer.
- Fallback: If local context fails, it initiates a live DuckDuckGo web search.
"""

import fitz
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS


# ==========================================
# INGESTION
# ==========================================
def ingest(pdf_path, bi_encoder):
    """
    Read a PDF, apply recursive splitting, embed, and store in ChromaDB.
    (Shares the same db logic as Level 3)
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

    client = chromadb.PersistentClient(path="./scholar_db_level5")
    try:
        client.delete_collection(name="scholar_rag_v5")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="scholar_rag_v5")
    chunk_ids = [f"id{i}" for i in range(len(my_chunks))]
    collection.add(documents=my_chunks, embeddings=embeddings, ids=chunk_ids)

    return client, collection, len(my_chunks)


def load_existing_db():
    try:
        client = chromadb.PersistentClient(path="./scholar_db_level5")
        collection = client.get_collection(name="scholar_rag_v5")
        if collection.count() > 0:
            return client, collection
    except Exception:
        pass
    return None


# ==========================================
# AGENTIC TOOLS
# ==========================================
def route_query(user_query, llm):
    prompt = f"""Classify input into exactly one category: GREETING or RESEARCH.
    Input: "{user_query}"
    Output ONLY the category name."""
    response = llm.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip().upper()


def expand_query(original_query, llm):
    prompt = f"""Generate 3 highly targeted, short search queries (under 6 words each) to find info related to: "{original_query}".
    Output exactly 3 queries, one per line. No intro text."""
    response = llm.generate_content(prompt, generation_config={"temperature": 0.2})
    return [original_query] + [q.strip() for q in response.text.split('\n') if q.strip()]


def grade_context(question, context, llm):
    prompt = f"""You are a strict grader evaluating document retrieval.
    Does the provided context contain the specific facts needed to answer the user's question?
    
    Context: {context}
    Question: {question}
    
    Output exactly 'YES' or 'NO' and nothing else."""
    response = llm.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip().upper()


def web_search(query):
    results = DDGS().text(query, max_results=3)
    web_context = ""
    for r in results:
        web_context += f"Source: {r['href']}\nSnippet: {r['body']}\n\n"
    return web_context


# ==========================================
# QUERY
# ==========================================
def query(user_question, collection, bi_encoder, cross_encoder, llm):
    """
    Run the Self-Healing RAG pipeline.
    Returns a dict with the answer and all intermediate debug data.
    """
    # 1. Route intent
    route = route_query(user_question, llm)

    if "GREETING" in route:
        return {
            "answer": "Hello! I am your Advanced RAG Agent. I will search our local database first, and the live internet if needed. What do you want to know?",
            "route": "GREETING",
            "expanded_queries": [],
            "bi_encoder_chunks": [],
            "cross_encoder_scores": [],
            "final_chunks": [],
            "grade": None,
            "healed": False,
        }

    # 2. Local DB Search
    search_queries = expand_query(user_question, llm)
    query_embeddings = bi_encoder.encode(search_queries).tolist()
    results = collection.query(query_embeddings=query_embeddings, n_results=5)

    unique_chunks = set()
    for document_list in results['documents']:
        for document in document_list:
            unique_chunks.add(document)
    retrieved_chunks = list(unique_chunks)

    # 3. Cross-Encoder re-ranking
    sentence_pairs = [[user_question, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(sentence_pairs)

    scored_chunks = list(zip([float(s) for s in scores], retrieved_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    best_chunks = [chunk for score, chunk in scored_chunks[:4]]
    local_context = "\n\n".join(best_chunks)

    # 4. Grading
    grade = grade_context(user_question, local_context, llm)
    healed = False

    if "YES" in grade:
        final_context = local_context
    else:
        # 5. Healing (Web Search)
        healed = True
        final_context = web_search(user_question)

    # 6. Synthesize
    prompt = f"""
    You are an expert research assistant. Answer the user's question using ONLY the provided context.
    If the context is from the web, integrate the URLs into your answer as citations.

    Context:
    {final_context}

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
        "final_chunks": best_chunks if not healed else final_context.split('\n\n')[:-1], # Approximate chunks for web
        "grade": grade,
        "healed": healed,
        "web_context": final_context if healed else None,
    }