import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# INITIALIZATION & SECURITY
# ==========================================
# Load the API key from the hidden .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("🚨 GEMINI_API_KEY not found! Please check your .env file.")

# Configure Gemini
genai.configure(api_key=api_key)
llm = genai.GenerativeModel('gemini-2.5-flash')

# ==========================================
# STEP 1: INGESTION & CHUNKING
# ==========================================
print("Reading the PDF...")
doc = fitz.open("sample.pdf") # Ensure 'sample.pdf' is in the same directory
full_text = ""
for page in doc:
    full_text += page.get_text()

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap) 
    return chunks

my_chunks = chunk_text(full_text)
print(f"Success! The paper was split into {len(my_chunks)} chunks.")

# ==========================================
# STEP 2: EMBEDDING
# ==========================================
print("\nLoading embedding model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Generating embeddings for all {len(my_chunks)} chunks...")
embeddings = model.encode(my_chunks)

# ==========================================
# STEP 3: VECTOR DATABASE STORAGE
# ==========================================
print("\nSetting up Vector Database...")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="scholar_rag_collection")

chunk_ids = [f"id{i}" for i in range(len(my_chunks))]

print("Inserting data into ChromaDB...")
collection.add(
    documents=my_chunks,      
    embeddings=embeddings,    
    ids=chunk_ids             
)
print(f"Successfully stored {collection.count()} chunks in the database!")

# ==========================================
# STEP 4: INTERACTIVE CHAT LOOP (RAG in Action)
# ==========================================
print("\n" + "="*50)
print("🤖 ScholarRAG is ready! Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    # 1. Get user input
    user_question = input("\nAsk a question about the paper: ")
    
    # Exit condition
    if user_question.lower() in ['exit', 'quit']:
        print("Shutting down ScholarRAG. Goodbye!")
        break

    print("Searching documents...")

    # 2. Embed the question and search ChromaDB
    question_embedding = model.encode([user_question])
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3 # Grab the top 3 most relevant chunks
    )

    # 3. Extract the text of those chunks
    retrieved_chunks = results['documents'][0]
    context_string = "\n\n".join(retrieved_chunks)

    # 4. Create the strict Prompt Template
    prompt = f"""
    You are an expert research assistant. Your job is to answer the user's question using ONLY the provided context from a research paper.

    If the answer is not contained in the context below, you must strictly reply with: "I'm sorry, but I cannot answer this based on the provided documents." Do not use outside knowledge.

    Context:
    {context_string}

    User Question:
    {user_question}
    """

    # 5. Generate and print the answer
    print("Synthesizing answer...")
    response = llm.generate_content(prompt)
    
    print("\n--- Answer ---")
    print(response.text)
    print("-" * 14)