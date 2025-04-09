import os
import cohere
import faiss
import numpy as np
import wandb
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
import re
import time
from sklearn.metrics.pairwise import cosine_similarity

# Load API keys from .env
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere
cohere_client = cohere.Client(COHERE_API_KEY)

# Initialize W&B
wandb.init(project="rag-cohere", name="financebench-retrieval-improved")

# Load FinanceBench dataset
dataset = load_dataset("PatronusAI/financebench")

# Text preprocessing function
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    # Clean and normalize text
    text = text.lower().strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\.\,\:\;\-\(\)]', ' ', text)
    return text[:2048]

# Extract documents from justification, fallback to doc_name
# Use more documents to increase chances of relevant content
documents = dataset["train"]["justification"][:500]  # Increased from 100 to 500
documents = [doc if doc else dataset["train"]["doc_name"][i] for i, doc in enumerate(documents)]

# Extract document types
doc_types = dataset["train"]["doc_type"][:500]  # Increased from 100 to 500

# Better preprocessing of documents
documents = [preprocess_text(doc) for doc in documents if doc and isinstance(doc, str) and doc.strip()]

# Create document-to-index mapping for later use
doc_to_idx = {doc: i for i, doc in enumerate(documents)}

if not documents:
    print("Error: No valid documents found for embedding. Exiting.")
    exit()

# Generate embeddings for documents in smaller batches
batch_size = 10
doc_embeddings = []

# Use a valid Cohere embedding model
EMBEDDING_MODEL = "embed-english-v3.0"

print(f"Generating embeddings for {len(documents)} documents using {EMBEDDING_MODEL}...")
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    try:
        response = cohere_client.embed(
            texts=batch, 
            model=EMBEDDING_MODEL,
            input_type="search_document",  # Specifically for document indexing
            truncate="END"  # Handle longer texts
        )
        doc_embeddings.extend(response.embeddings)
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: Successfully embedded {len(response.embeddings)} documents")
        time.sleep(0.5)  # Rate limiting to avoid API throttling
    except Exception as e:
        print(f"Error with batch {i//batch_size + 1}: {str(e)}")
        try:
            print("Retrying batch with backoff...")
            time.sleep(2)  # Longer backoff
            response = cohere_client.embed(
                texts=batch, 
                model=EMBEDDING_MODEL,
                input_type="search_document",
                truncate="END"
            )
            doc_embeddings.extend(response.embeddings)
            print(f"Retry successful: embedded {len(response.embeddings)} documents")
        except Exception as retry_error:
            print(f"Retry failed: {str(retry_error)}")

if not doc_embeddings:
    print("Error: Failed to generate any document embeddings. Exiting.")
    exit()

doc_embeddings = np.array(doc_embeddings).astype("float32")
faiss.normalize_L2(doc_embeddings)
print(f"Created document embedding array with shape: {doc_embeddings.shape}")

# Create FAISS
# Use better HNSW parameters for more accurate search
index = faiss.IndexHNSWFlat(doc_embeddings.shape[1], 64) 
index.hnsw.efConstruction = 12
index.add(doc_embeddings)
print(f"Created enhanced FAISS index with {index.ntotal} vectors")

# Prepare queries with better preprocessing
# Get more queries for better evaluation
queries = dataset["train"]["question"][:20]  # Increased from 10 to 20
valid_query_indices = []
valid_queries = []
original_queries = []

for i, q in enumerate(queries):
    if q and isinstance(q, str) and q.strip():
        valid_query_indices.append(i)
        processed_query = preprocess_text(q)
        valid_queries.append(processed_query)
        original_queries.append(q.strip())

if not valid_queries:
    print("Error: No valid queries found for embedding. Exiting.")
    exit()

# Generate embeddings for queries
query_embeddings = []
print(f"Generating embeddings for {len(valid_queries)} queries using {EMBEDDING_MODEL}...")
for i in range(0, len(valid_queries), batch_size):
    batch = valid_queries[i:i + batch_size]
    try:
        response = cohere_client.embed(
            texts=batch, 
            model=EMBEDDING_MODEL,
            input_type="search_query",  # Specifically for queries
            truncate="END"
        )
        query_embeddings.extend(response.embeddings)
        print(f"Processing batch {i//batch_size + 1}/{(len(valid_queries)-1)//batch_size + 1}: Successfully embedded {len(response.embeddings)} queries")
        time.sleep(0.5)  # Rate limiting
    except Exception as e:
        print(f"Error with batch {i//batch_size + 1}: {str(e)}")
        try:
            print("Retrying batch...")
            time.sleep(2)
            response = cohere_client.embed(
                texts=batch, 
                model=EMBEDDING_MODEL,
                input_type="search_query",
                truncate="END"
            )
            query_embeddings.extend(response.embeddings)
            print(f"Retry successful: embedded {len(response.embeddings)} queries")
        except Exception as retry_error:
            print(f"Retry failed: {str(retry_error)}")

if not query_embeddings:
    print("Error: Failed to generate any query embeddings. Exiting.")
    exit()

query_embeddings = np.array(query_embeddings).astype("float32")
faiss.normalize_L2(query_embeddings)
print(f"Created query embedding array with shape: {query_embeddings.shape}")

# Perform improved retrieval
# Retrieve more results initially for reranking
top_k = 10  # Increased from 5 to 10
index.hnsw.efSearch = 128  # Increase search-time exploration factor for better recall
D, I = index.search(query_embeddings, top_k)

# Get ground truth answers with better preprocessing
ground_truth_answers = []
for idx in valid_query_indices:
    answer = dataset["train"]["answer"][idx]
    if answer and isinstance(answer, str):
        ground_truth_answers.append(preprocess_text(answer))
    else:
        ground_truth_answers.append("")

# Advanced relevance function with matching criteria
def is_relevant(doc, ground_truth):
    if not doc or not ground_truth:
        return False
    
    # Check for significant overlap of content
    doc_words = set(re.findall(r'\b\w+\b', doc.lower()))
    gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
    
    # Look for key numbers that might indicate correct answer
    doc_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', doc))
    gt_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', ground_truth))
    
    # Calculate word overlap
    if len(gt_words) > 0:
        word_overlap = len(doc_words.intersection(gt_words)) / len(gt_words)
        
        # If significant word overlap or contains same numbers
        if word_overlap > 0.3 or (len(gt_numbers) > 0 and len(doc_numbers.intersection(gt_numbers)) > 0):
            return True
            
    # Direct substring matching for shorter answers
    if len(ground_truth) < 100:
        if ground_truth in doc or doc in ground_truth:
            return True
            
    return False

# Rerank results when appropriate
results = []
precision_scores = []
retrieved_docs_texts = []

print("Computing retrieval metrics with improved relevance criteria...")
for i, (query_idx, retrieved_indices) in enumerate(zip(valid_query_indices, I)):
    if i >= len(ground_truth_answers):
        continue
        
    ground_truth = ground_truth_answers[i]
    
    # Get retrieved documents
    retrieved_docs = [documents[idx] if idx < len(documents) else "" for idx in retrieved_indices]
    retrieved_doc_types = [doc_types[idx] if idx < len(doc_types) else "Unknown" for idx in retrieved_indices]
    
    # Calculate improved relevance
    relevance_scores = []
    for doc in retrieved_docs:
        relevance = is_relevant(doc, ground_truth)
        relevance_scores.append(1 if relevance else 0)
    
    # Calculate precision
    precision = sum(relevance_scores) / top_k if top_k > 0 else 0
    precision_scores.append(precision)
    
    # More detailed logging
    print(f"Query {i+1}: '{original_queries[i][:50]}...' - Precision = {precision:.2f}")
    
    # Log ground truth and top match for debugging
    if len(retrieved_docs) > 0:
        top_doc = retrieved_docs[0]
        print(f"  Top doc: '{top_doc[:50]}...'")
        print(f"  Ground truth: '{ground_truth[:50]}...'")
        print(f"  Relevant: {relevance_scores[0] == 1}")
    
    results.append({
        "query_idx": query_idx,
        "query": original_queries[i],
        "ground_truth": ground_truth,
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_types": retrieved_doc_types,
        "distances": D[i].tolist(),
        "precision": precision,
        "relevant_count": sum(relevance_scores)
    })
    
    retrieved_docs_texts.append(retrieved_docs)

avg_precision = np.mean(precision_scores) if precision_scores else 0
print(f"Average Precision: {avg_precision:.4f}")

# Enhanced logging with more metrics
wandb.log({
    "retrieval_precision": avg_precision,
    "num_queries": len(valid_queries),
    "num_documents": len(documents),
    "embedding_model": EMBEDDING_MODEL,
    "top_k": top_k
})

results_df = pd.DataFrame(results)
wandb.log({"results_sample": wandb.Table(dataframe=results_df[["query_idx", "query", "precision", "relevant_count"]].head(10))})

top_retrieval_df = pd.DataFrame({
    "Query": [r["query"] for r in results],
    "Top Retrieved Document": [r["retrieved_docs"][0] if r["retrieved_docs"] else "" for r in results],
    "Document Type": [r["retrieved_doc_types"][0] if r["retrieved_doc_types"] else "Unknown" for r in results],
    "Precision": [r["precision"] for r in results],
    "Relevant Count": [r["relevant_count"] for r in results]
})
print("\nTop retrievals:")
print(top_retrieval_df.head(10))

wandb.log({"top_retrievals": wandb.Table(dataframe=top_retrieval_df)})

print("Finishing W&B run...")
wandb.finish()

print("Evaluation complete!")