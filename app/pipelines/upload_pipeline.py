# app/utils/upload_pipeline.py
import os, uuid, hashlib
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.rag_config import (
    QDRANT_URL, QDRANT_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL_TYPE, LOCAL_EMBEDDING_MODEL_NAME, HUGGINGFACE_EMBEDDING_MODEL_NAME
)

# 1. Khởi client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 2. TextSplitter & Embedder
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
def get_embedder():
    if EMBEDDING_MODEL_TYPE == "local":
        return SentenceTransformerEmbeddings(model_name=LOCAL_EMBEDDING_MODEL_NAME)
    else:
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME)
embedder = get_embedder()

# 3. CRUD collection
def create_collection(name: str, vector_size: int = 768, distance: str = "COSINE"):
    # nếu đã tồn tại sẽ bắn exception
    try:
        client.get_collection(collection_name=name)
        raise ValueError(f"Collection `{name}` đã tồn tại.")
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance[distance.upper()]),
        )

def delete_collection(name: str):
    client.delete_collection(collection_name=name)

# 4. Hàm ingest file
def ingest_file(
    file_path: Path,
    document_type: str,
    collection_name: str,
    use_stable_id: bool = True
):
    # 4.1 chọn loader
    ext = file_path.suffix.lower()
    if ext == ".txt":
        loader = TextLoader(str(file_path))
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(str(file_path))
    elif ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        raise ValueError(f"Không hỗ trợ định dạng `{ext}`")

    # 4.2 load & split
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    points = []
    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
    for idx, chunk in enumerate(chunks):
        if use_stable_id:
            # ID ổn định để upsert đè lên bản cũ
            id_ = hashlib.md5(f"{file_hash}-{idx}".encode()).hexdigest()
        else:
            id_ = str(uuid.uuid4())
        vec = embedder.embed_query(chunk.page_content)
        points.append(
            PointStruct(
                id=id_,
                vector=vec,
                payload={
                    "content": chunk.page_content,
                    "metadata": {
                        "document_type": document_type,
                        "source_file": file_path.name,
                        "file_hash": file_hash
                    }
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)
