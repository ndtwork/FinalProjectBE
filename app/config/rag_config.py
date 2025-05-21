# app/config/rag_config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Tải các biến từ file .env

# Cấu hình chung của ứng dụng RAG
APP_NAME = os.getenv("APP_NAME", "RAGChatApp")

# Cấu hình lựa chọn LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Chọn 'ollama' hoặc 'huggingface' để thay đổi nhà cung cấp LLM

# --- Cấu hình cho Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Địa chỉ server Ollama
LLM_MODEL_NAME_OLLAMA = os.getenv("LLM_MODEL_NAME_OLLAMA", "llama2")  # Tên model bạn đã tải về trong Ollama

# --- Cấu hình cho Hugging Face Transformers ---
LLM_MODEL_HUGGINGFACE = os.getenv("LLM_MODEL_HUGGINGFACE", "VietAI/vit5-base")  # Tên model từ Hugging Face Hub

# --- Cấu hình cho Embedding Model ---
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "local")  # Chọn 'local' hoặc 'huggingface'
# Cấu hình cho Embedding Model Local (ví dụ: Sentence Transformers)
LOCAL_EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Cấu hình cho Embedding Model từ Hugging Face Hub
HUGGINGFACE_EMBEDDING_MODEL_NAME = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Cấu hình cho Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "quy_che")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Cấu hình khác cho RAG pipeline ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Bạn có thể thêm các cấu hình khác nếu cần