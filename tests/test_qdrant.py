# app/pipelines/upload_rag.py

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_document_loaders import TextLoader, MarkdownLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load biến môi trường
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")

QDRANT_URL          = os.getenv("QDRANT_URL")
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME     = os.getenv("QDRANT_COLLECTION_NAME", "quy_che_full")
EMBEDDING_MODEL_NAME= os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", 400))
VECTOR_SIZE         = 128  # phải khớp dimension của EMBEDDING_MODEL_NAME

# 2. Khởi tạo client & embedder (chạy 1 lần)
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# Tạo collection nếu chưa tồn tại
if not client.collections_api.exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"[INFO] Đã tạo mới collection `{COLLECTION_NAME}` trên Qdrant.")
else:
    print(f"[INFO] Collection `{COLLECTION_NAME}` đã tồn tại, sẽ upsert thêm dữ liệu.")

embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}  # hoặc "cuda" nếu có GPU
    # , "trust_remote_code": True  # nếu model custom cần
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def upload_document_to_qdrant(file_path: str, document_type: str, collection_name: str):
    """
    Tách file thành chunks, tính embedding và upsert vào Qdrant.
    """
    # 1. Chọn loader dựa theo extension
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        loader = TextLoader()
    elif ext == ".md":
        loader = MarkdownLoader()
    elif ext == ".pdf":
        loader = PyPDFLoader()
    else:
        print(f"[WARN] Loại file {ext} chưa được hỗ trợ: bỏ qua {file_path}")
        return

    # 2. Load tài liệu ⇒ List[Document]
    documents = loader.load(file_path=file_path)
    # 3. Chia thành chunks
    chunks = text_splitter.split_documents(documents)

    # 4. Chuẩn bị list of PointStruct
    points = []
    for chunk in chunks:
        vector = embeddings_model.embed_query(chunk.page_content)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # id duy nhất
                vector=vector,
                payload={
                    "content": chunk.page_content,
                    "metadata": {"document_type": document_type, "source_file": Path(file_path).name}
                }
            )
        )

    # 5. Upsert vào Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Upsert {len(points)} chunks từ `{Path(file_path).name}` vào collection `{collection_name}`.")

if __name__ == "__main__":
    data_dir = project_root / "data"
    regulations_dir = data_dir / "regulations"
    faq_dir         = data_dir / "faq"
    related_dir     = data_dir / "related_issues"

    # Đảm bảo các folder data tồn tại (nếu cần, có thể tự tạo tương tự)
    for folder in (regulations_dir, faq_dir, related_dir):
        if not folder.exists():
            print(f"[WARN] Thư mục không tồn tại: {folder}, bỏ qua.")
            continue

        # Duyệt lần lượt từng folder và gọi hàm upload
        if folder == regulations_dir:
            doc_type = "Regulation"
        elif folder == faq_dir:
            doc_type = "FAQ"
        else:
            doc_type = "RelatedIssue"

        for file_name in os.listdir(folder):
            file_path = folder / file_name
            upload_document_to_qdrant(str(file_path), document_type=doc_type, collection_name=COLLECTION_NAME)

    print("🎉 Hoàn tất quá trình tải dữ liệu lên Qdrant.")
