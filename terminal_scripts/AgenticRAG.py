import os
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# INITIALIZATION & SECURITY
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("🚨 GEMINI_API_KEY not found! Please check your .env file.")

genai.configure(api_key=api_key)
llm = genai.GenerativeModel('gemini-2.5-flash')

# ==========================================
# INGESTION & STORAGE (Level 3 Setup)
# ==========================================
print("\nConnecting to Vector Database...")
# Dedicated Level 3 database folder
chroma_client = chromadb.PersistentClient(path="./scholar_db_level3")
collection = chroma_client.get_or_create_collection(name="scholar_rag_v3")

if collection.count() == 0:
    print("Database is empty! Starting the one-time ingestion process...")
    print("Reading the PDF...")
    doc = fitz.open("sample.pdf") 
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""] 
    )

    my_chunks = []
    for page in doc:
        my_chunks.extend(text_splitter.split_text(page.get_text()))

    print(f"Success! The paper was intelligently split into {len(my_chunks)} chunks.")
    
    print("\nLoading Bi-Encoder (this may take a moment)...")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(my_chunks)} chunks...")
    embeddings = bi_encoder.encode(my_chunks)
    
    chunk_ids = [f"id{i}" for i in range(len(my_chunks))]
    print("Saving data to hard drive...")
    collection.add(documents=my_chunks, embeddings=embeddings, ids=chunk_ids)
    print("Ingestion complete!")

else:
    print(f"✅ Database already exists on disk with {collection.count()} chunks.")
    print("Loading Bi-Encoder...")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading Cross-Encoder (Deep Scoring Re-ranker)...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# AGENTIC ROUTER & QUERY EXPANSION
# ==========================================
def route_query(user_query):
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

def expand_query(original_query):
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
# STEP 4: INTERACTIVE CHAT LOOP 
# ==========================================
print("\n" + "="*50)
print("🤖 ScholarRAG (Level 3 - Agentic) is ready! Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    user_question = input("\nAsk a question about the paper: ")
    
    if user_question.lower() in ['exit', 'quit']:
        print("Shutting down ScholarRAG. Goodbye!")
        break

    # --- THE ROUTER ---
    print("\n🧠 Agent is analyzing intent...")
    route = route_query(user_question)
    
    if "GREETING" in route:
        print("🔀 Route selected: Casual Conversation (Skipping Vector DB)")
        print("\n--- Answer ---")
        print("Hello! I am ScholarRAG, an autonomous research assistant. Ask me anything about the loaded document!")
        print("-" * 14)
        continue 
        
    print("🔀 Route selected: Document Research (Engaging RAG Pipeline)")

    # 1. Expand the query
    print("\nExpanding your query...")
    search_queries = expand_query(user_question)
    query_embeddings = bi_encoder.encode(search_queries)
    
    # 2. FAST BI-ENCODER RETRIEVAL (Casting a wide net)
    print("Executing wide vector search...")
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=5 
    )

    unique_chunks = set()
    for document_list in results['documents']:
        for document in document_list:
            unique_chunks.add(document)
            
    retrieved_chunks = list(unique_chunks)
    print(f"📚 Bi-Encoder retrieved {len(retrieved_chunks)} total chunks.")

    # 3. CROSS-ENCODER RE-RANKING
    print("⚖️ Re-ranking chunks with Cross-Encoder...")
    sentence_pairs = [[user_question, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(sentence_pairs)
    
    scored_chunks = list(zip(scores, retrieved_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # THROW AWAY THE TRASH: Keep only the absolute best 4 chunks
    best_chunks = [chunk for score, chunk in scored_chunks[:4]]
    print(f"✅ Filtered down to the top {len(best_chunks)} highest-quality chunks.")
    
    # 4. SYNTHESIS
    context_string = "\n\n".join(best_chunks)
    
    prompt = f"""
    You are an expert research assistant. Your job is to answer the user's question using ONLY the provided context from a research paper.

    If the answer is not contained in the context below, you must strictly reply with: "I'm sorry, but I cannot answer this based on the provided documents." Do not use outside knowledge.

    Context:
    {context_string}

    User Question:
    {user_question}
    """

    print("Synthesizing answer...")
    response = llm.generate_content(prompt)
    
    print("\n--- Answer ---")
    print(response.text)
    print("-" * 14)