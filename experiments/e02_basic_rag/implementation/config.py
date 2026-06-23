"""E02 / E02.1 / E02.2 — Basic, Sparse, and Hybrid RAG configuration."""
EXPERIMENT_ID   = "E02"
SYSTEM_TYPE     = "Basic RAG"
LLM_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB       = "ChromaDB"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 3
RRF_K           = 60   # Reciprocal Rank Fusion constant for E02.2
