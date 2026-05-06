"""
Level 4: RAG Evaluation Pipeline
----------------------------------
Core logic extracted from terminal_scripts/EvaluateRAG.py.
- Uses the Level 3 database
- 15 predefined ground-truth Q&A pairs
- Ragas framework for automated evaluation
- Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy


# ==========================================
# GROUND TRUTH DATASET
# ==========================================
EVAL_DATA = [
    {
        "question": "What are two key benefits of Conversational agents compared to traditional enterprise systems?",
        "ground_truth": "They enable users to pose questions using natural language, and they are increasingly capable of tackling complex search tasks to facilitate problem-solving."
    },
    {
        "question": "How is hallucination defined in the context of large language models?",
        "ground_truth": "Hallucination is defined as content that is inconsistent with real-world facts or user inputs."
    },
    {
        "question": "What is the core idea of Retrieval-Augmented Generation (RAG)?",
        "ground_truth": "The core idea is to combine the generative capabilities of large language models with external knowledge retrieved from a separate database."
    },
    {
        "question": "What does parametric memory refer to?",
        "ground_truth": "Parametric memory refers to information that is stored in the parameters of a model, which can be used later to regenerate information."
    },
    {
        "question": "What is non-parametric memory?",
        "ground_truth": "Non-parametric memory refers to information that is outside of a model, such as domain-specific data from an external database or internet sites like Wikipedia."
    },
    {
        "question": "What are the three fundamental parts of a plain vanilla RAG pipeline?",
        "ground_truth": "The three fundamental parts are retrieval, augmentation, and generation."
    },
    {
        "question": "What is the purpose of the embedding model in a RAG architecture?",
        "ground_truth": "The embedding model translates data of different modalities into a vector, which is then used to measure the similarity between a query and documents in the database."
    },
    {
        "question": "How does the retriever select the most relevant information?",
        "ground_truth": "The retriever calculates the similarity score between the vector of an input query and the vectors of documents in the database, typically selecting the top hits ranked by this score."
    },
    {
        "question": "What does RAPTOR stand for and what does it do?",
        "ground_truth": "RAPTOR stands for Recursive Abstractive Processing for Tree-Organized Retrieval, and it recursively embeds, clusters, and summarizes text at multiple levels of abstraction."
    },
    {
        "question": "How does GraphRAG improve RAG-based tasks?",
        "ground_truth": "GraphRAG extracts knowledge graphs from text and structures them hierarchically to leverage these graph-based structures for improved retrieval."
    },
    {
        "question": "What is RAFT and why is it useful?",
        "ground_truth": "RAFT stands for retrieval-augmented fine-tuning, which combines RAG and fine-tuning by creating synthetic datasets to fine-tune models to specific domains, leading to better performance in specialized fields like medicine."
    },
    {
        "question": "What does the concept of 'grounding' refer to in RAG?",
        "ground_truth": "Grounding refers to providing valid references or links to the contextual data stored in the vector database, allowing users to verify where the generated information comes from."
    },
    {
        "question": "What is the 'blinkered chunk effect' (BCE)?",
        "ground_truth": "The blinkered chunk effect refers to the limitation where a single extracted text chunk lacks comprehensive understanding because the broader context of the entire document is missing."
    },
    {
        "question": "Why does RAG present new challenges for data management?",
        "ground_truth": "RAG requires additional effort to merge dynamically changing data from heterogeneous sources, requiring organizations to allocate more resources to ensure high data quality and eliminate false information."
    },
    {
        "question": "How does RAG solve the problem of foundation models having outdated training data?",
        "ground_truth": "RAG allows more recent data to be added as non-parametric memory, enabling the system to retrieve accurate and up-to-date information that was not present during the model's initial training."
    }
]


# ==========================================
# EVALUATION RUNNER
# ==========================================
def run_evaluation(collection, bi_encoder, cross_encoder, rag_llm, api_key, progress_callback=None):
    """
    Run the full Ragas evaluation pipeline against the Level 3 database.
    progress_callback(current, total, message) is called for UI updates.
    Returns (overall_scores_dict, per_question_results_list).
    """
    # Setup Ragas judge models
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_emb = LangchainEmbeddingsWrapper(judge_embeddings)

    samples = []
    per_question = []
    total = len(EVAL_DATA)

    for i, item in enumerate(EVAL_DATA):
        user_query = item["question"]
        if progress_callback:
            progress_callback(i, total, f"Testing: {user_query[:60]}...")

        # Fast retrieval
        query_emb = bi_encoder.encode([user_query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=5)

        unique_chunks = set()
        for doc_list in results['documents']:
            for doc in doc_list:
                unique_chunks.add(doc)
        retrieved_chunks = list(unique_chunks)

        # Re-ranking
        if retrieved_chunks:
            sentence_pairs = [[user_query, chunk] for chunk in retrieved_chunks]
            scores = cross_encoder.predict(sentence_pairs)
            scored_chunks = list(zip(scores, retrieved_chunks))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            best_chunks = [chunk for score, chunk in scored_chunks[:4]]
        else:
            best_chunks = []

        # Synthesis
        context_string = "\n\n".join(best_chunks)
        prompt = f"Answer strictly using context. Context: {context_string} Question: {user_query}"
        response = rag_llm.generate_content(prompt).text

        per_question.append({
            "question": user_query,
            "ground_truth": item["ground_truth"],
            "response": response,
            "num_chunks": len(best_chunks),
        })

        sample = SingleTurnSample(
            user_input=user_query,
            retrieved_contexts=best_chunks,
            response=response,
            reference=item["ground_truth"]
        )
        samples.append(sample)

    if progress_callback:
        progress_callback(total, total, "Running Ragas metrics (this may take a few minutes)...")

    dataset = EvaluationDataset(samples=samples)
    metrics_to_run = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]

    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    return dict(results), per_question
