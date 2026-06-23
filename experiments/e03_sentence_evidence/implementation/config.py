"""E03 — RAG with Sentence Evidence configuration."""
EXPERIMENT_ID   = "E03"
SYSTEM_TYPE     = "RAG with Sentence Evidence"
LLM_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB       = "ChromaDB"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 3
TOP_N_SENTENCES = 5
