# app/pipelines/upload_rag.py

import os
import uuid
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import time  # <- thêm import time để đo thời gian


# ---- 1. Load biến môi trường từ file .env ----
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")

QDRANT_URL          = os.getenv("QDRANT_URL")
QDRANT_API_KEY      = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME     = "quy_che_full" #os.getenv("quy_che_full")
EMBEDDING_MODEL_NAME= os.getenv("LOCAL_EMBEDDING_MODEL_NAME")
CHUNK_SIZE          =  800 #int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP       =  400 #int(os.getenv("CHUNK_OVERLAP", 400))

# ---- 2. Khởi tạo QdrantClient & tạo (nếu chưa có) collection ----
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout = 60.0,
    prefer_grpc=False  # hoặc True nếu bạn ưu tiên gRPC
)

# Nếu collection chưa tồn tại, tạo mới
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"[INFO] Collection `{COLLECTION_NAME}` đã tồn tại.")
except Exception:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"[INFO] Đã tạo mới collection `{COLLECTION_NAME}`")


# ---- 3. Khởi tạo Embedder (dùng HuggingFaceEmbeddings thay cho SentenceTransformerEmbeddings) ----
# Nếu trước đây bạn dùng SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME),
# thì giờ đổi thành:
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}  # hoặc "cuda" nếu bạn có GPU và torch hỗ trợ
    # nếu EMBEDDING_MODEL_NAME cần custom pooling (SimCSE), thêm "trust_remote_code": True
)

# ---- 4. Text splitter (không thay đổi) ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def upload_document_to_qdrant(
    file_path: str,
    document_type: str,
    collection_name: str
):
    """
    Chia file thành từng đoạn (chunk), tính embedding, rồi upsert vào Qdrant.
    file_path: đường dẫn tới file (txt/md/pdf).
    document_type: metadata để phân biệt (Regulation, FAQ, RelatedIssue, v.v.).
    collection_name: tên collection Qdrant.
    """
    # 4.1. Chọn loader dựa trên đuôi file
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        loader = TextLoader(file_path=file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path=file_path)
    else:
        print(f"[WARN] Loại file {ext} chưa hỗ trợ: bỏ qua {file_path}")
        return

    # 4.2. Load toàn bộ Document từ file
    documents = loader.load()

    # 4.3. Chia thành chunks
    start_split = time.time()  # BẮT ĐẦU đo thời gian tách chunk
    chunks = text_splitter.split_documents(documents)
    t_split = time.time() - start_split  # KẾT THÚC đo
    print(f"[METRIC] Tách {len(chunks)} chunk mất {t_split:.4f} s")  # IN kết quả đo

    # 4.4. Chuẩn bị list PointStruct để upsert
    points = []
    start_embed = time.time()  # BẮT ĐẦU đo thời gian tính embedding
    for chunk in chunks:
        vector = embeddings_model.embed_query(chunk.page_content)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),   # tạo ID duy nhất cho mỗi chunk
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
    t_embed = time.time() - start_embed  # KẾT THÚC đo
    print(f"[METRIC] Tính embedding cho {len(chunks)} chunk mất {t_embed:.4f} s")  # IN kết quả đo

    # 4.5. Upsert lên Qdrant
    start_upsert = time.time()  # BẮT ĐẦU đo thời gian upsert
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    t_upsert = time.time() - start_upsert  # KẾT THÚC đo
    print(f"[METRIC] Upsert {len(points)} chunk mất {t_upsert:.4f} s")  # IN kết quả đo

    print(f"[OK] Đã upsert {len(points)} chunks từ `{Path(file_path).name}` vào `{collection_name}`.")

if __name__ == "__main__":
    # ---- 5. Xác định các thư mục dữ liệu và gọi hàm upload ----
    data_dir = project_root / "data"
    regulations_dir    = data_dir / "regulations"
    faq_dir            = data_dir / "faq"
    related_issues_dir = data_dir / "related_issues"

    # Nếu chưa có thư mục, bạn có thể tạo bằng tay hoặc tự động tạo:
    for folder in (regulations_dir, faq_dir, related_issues_dir):
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"[WARN] Tạo mới thư mục: {folder} (hiện đang rỗng)")

    # 5.1. Process Regulations (txt, md, pdf)
    if regulations_dir.exists():
        for filename in os.listdir(regulations_dir):
            file_path = regulations_dir / filename
            if file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
                upload_document_to_qdrant(
                    str(file_path),
                    document_type="Regulation",
                    collection_name=COLLECTION_NAME
                )
    else:
        print(f"[ERROR] Thư mục không tồn tại: {regulations_dir}")

    # 5.2. Process FAQs
    if faq_dir.exists():
        for filename in os.listdir(faq_dir):
            file_path = faq_dir / filename
            if file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
                upload_document_to_qdrant(
                    str(file_path),
                    document_type="FAQ",
                    collection_name=COLLECTION_NAME
                )
    else:
        print(f"[ERROR] Thư mục không tồn tại: {faq_dir}")

    # 5.3. Process Related Issues
    if related_issues_dir.exists():
        for filename in os.listdir(related_issues_dir):
            file_path = related_issues_dir / filename
            if file_path.suffix.lower() in {".txt", ".md", ".pdf"}:
                upload_document_to_qdrant(
                    str(file_path),
                    document_type="RelatedIssue",
                    collection_name=COLLECTION_NAME
                )
    else:
        print(f"[ERROR] Thư mục không tồn tại: {related_issues_dir}")

    print("🎉 Hoàn tất quá trình upload dữ liệu lên Qdrant.")
