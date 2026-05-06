import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

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
# STEP 1 & 2 & 3: SMART INGESTION & STORAGE
# ==========================================
print("\nConnecting to Level 2 (Intermediate) Vector Database...")

# Note: We are using a fresh database path here to avoid mixing old, metadata-less chunks!
chroma_client = chromadb.PersistentClient(path="./scholar_db_intermediate")
collection = chroma_client.get_or_create_collection(name="scholar_rag_v2")

if collection.count() == 0:
    print("Database is empty! Starting the one-time ingestion process...")
    
    print("Reading the PDF and attaching Metadata...")
    doc = fitz.open("sample.pdf") 
    
    # Initialize the Smart Chunker
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""] 
    )

    my_chunks = []
    my_metadata = [] # --- UPGRADE 3: METADATA LIST ---

    # Instead of mashing text together, we process page by page
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        
        # Split just this page
        page_chunks = text_splitter.split_text(page_text)
        
        # Add the chunks to our master list
        my_chunks.extend(page_chunks)
        
        # Create a metadata tag for EVERY chunk on this page (e.g., {"page": 1})
        for _ in range(len(page_chunks)):
            my_metadata.append({"page": page_num + 1}) # +1 because page_num starts at 0

    print(f"Success! The paper was intelligently split into {len(my_chunks)} chunks.")
    
    print("\nLoading embedding model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Generating embeddings for {len(my_chunks)} chunks...")
    embeddings = model.encode(my_chunks)
    
    chunk_ids = [f"id{i}" for i in range(len(my_chunks))]
    print("Saving data to hard drive...")
    
    # Pass the metadata to ChromaDB!
    collection.add(
        documents=my_chunks, 
        embeddings=embeddings, 
        metadatas=my_metadata, 
        ids=chunk_ids
    )
    print("Ingestion complete!")

else:
    print(f"Database already exists on disk with {collection.count()} chunks. Skipping ingestion!")
    print("Loading embedding model for chat...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# UPGRADE 2: QUERY EXPANSION FUNCTION
# ==========================================
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
print("🤖 ScholarRAG (Level 2) is ready! Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    user_question = input("\nAsk a question about the paper: ")
    
    if user_question.lower() in ['exit', 'quit']:
        print("Shutting down ScholarRAG. Goodbye!")
        break

    # --- UPGRADE 3: METADATA FILTERING INPUT ---
    page_filter = input("Limit search to a specific page? (Enter number, or press Enter to search all pages): ")

    print("\nExpanding your query for better search results...")
    search_queries = expand_query(user_question)
    
    print(f"🔍 Searching database with {len(search_queries)} variations:")
    for q in search_queries:
        print(f"  - {q}")

    query_embeddings = model.encode(search_queries)
    
    # Build our search parameters
    search_params = {
        "query_embeddings": query_embeddings,
        "n_results": 2
    }
    
    # If the user typed a number, add the metadata filter!
    if page_filter.strip().isdigit():
        target_page = int(page_filter.strip())
        search_params["where"] = {"page": target_page}
        print(f"🎯 Applying strict metadata filter: Page {target_page}")

    # Execute the search with our dynamic parameters
    results = collection.query(**search_params)

    # Flatten and deduplicate
    unique_chunks = set()
    for document_list in results['documents']:
        for document in document_list:
            unique_chunks.add(document)
            
    retrieved_chunks = list(unique_chunks)
    context_string = "\n\n".join(retrieved_chunks)
    print(f"📚 Retrieved {len(retrieved_chunks)} unique context chunks.")

    # --- SYNTHESIS ---
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