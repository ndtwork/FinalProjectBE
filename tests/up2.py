from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_community.document_loaders import TextLoader, MarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os, uuid
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")

QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME      = "demo2"
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 400))

# Tăng timeout lên 60s
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,   # hoặc True nếu bạn có gRPC
    timeout=60.0
)

# Tạo hoặc recreate collection dimension 768 (tương ứng SimCSE-phobert)
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    client.delete_collection(collection_name=COLLECTION_NAME)
except:
    pass
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

def upload_file(file_path: str, document_type: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        loader = TextLoader(file_path=file_path)
    elif ext == ".md":
        loader = MarkdownLoader(file_path=file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path=file_path)
    else:
        print(f"[WARN] Loại file {ext} không hỗ trợ")
        return

    documents = loader.load()
    chunks = text_splitter.split_documents(documents)

    # Tạo list PointStruct
    points = []
    for chunk in chunks:
        vector = embeddings_model.embed_query(chunk.page_content)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content": chunk.page_content,
                    "metadata": {
                        "document_type": document_type,
                        "source_file": Path(file_path).name
                    }
                }
            )
        )

    # Upsert theo batch 20 point mỗi lần
    batch_size = 20
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )
        print(f"[OK] Upserted {len(batch)} chunks from {Path(file_path).name}")

if __name__ == "__main__":
    # Ví dụ upload một file PDF lớn
    # Cách 1: raw string
    upload_file(
        r"C:\Users\nguye\PycharmProjects\FinalProjectBE\data\regulations\Quy chế CTSV ĐHBK Hà Nội 2025.3.10_final.pdf",
        "Regulation"
    )

    # Kết cấu tương tự cho các file khác
