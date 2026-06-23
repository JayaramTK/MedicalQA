"""E04 — RAG with Uncertainty Handling configuration."""
EXPERIMENT_ID   = "E04"
SYSTEM_TYPE     = "RAG with Uncertainty Handling"
LLM_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB       = "ChromaDB"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 3
HIGH_CONF_THRESHOLD = 0.60
LOW_CONF_THRESHOLD  = 0.40
