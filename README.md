# rag-financebench-cohere
RAG Pipeline with Cohere Embeddings on FinanceBench Dataset for Financial Question Answering.

## What This Project Does

-  Retrieves financial justifications and documents relevant to natural language financial queries.
-  Uses Cohere's `embed-english-v3.0` model to convert both queries and documents into dense vector embeddings.
-  Leverages **FAISS** for fast and accurate similarity search.
-  Evaluates performance using custom precision metrics and logs results via **Weights & Biases**.

## Dataset
- **FinanceBench** by [PatronusAI](https://huggingface.co/datasets/PatronusAI/financebench)
- Contains questions, answers, justifications, and document metadata about financial filings and reports.

## Technologies Used

- Cohere APIfor embeddings
- FAISS for efficient vector similarity search
- W&B for experiment tracking and logging
- Python

  ## Features

-  Preprocesses and cleans financial documents and queries.
- Batches embedding calls with retry and backoff logic.
- Uses HNSW FAISS index with fine-tuned parameters for improved recall.
- Includes advanced relevance function to better match answers using both semantic and numeric content.
- Logs query-level precision scores and top document matches
