"""E05 — Final Combined RAG configuration."""
EXPERIMENT_ID   = "E05"
SYSTEM_TYPE     = "Final Combined RAG System"
LLM_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB       = "ChromaDB"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 5
TOP_N_SENTENCES = 7
HIGH_CONF_THRESHOLD = 0.60
LOW_CONF_THRESHOLD  = 0.40
QUERY_EXPANSIONS = {
    "symptoms": "signs manifestations",
    "treatment": "therapy management",
    "cause": "etiology risk factors",
    "prevent": "prevention reduce risk",
    "diagnose": "diagnosis detection screening",
}
