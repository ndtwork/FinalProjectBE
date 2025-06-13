#!/usr/bin/env python
"""
Đo Precision@3 cho 4 cấu hình chunk_size–overlap
mà KHÔNG phải đụng vào mã nguồn RAGPipeline gốc.

Cách chạy:
  python benchmark_precision.py --build    # ingest 4 collection (chạy 1 lần)
  python benchmark_precision.py --eval     # tính Precision@3
"""

import os, json, uuid, argparse, time, statistics
from pathlib import Path
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import rag_config as cfg  # lấy biến môi trường & model

# ---------- Cấu hình ----------
DATA_DIR      = Path("data")              # nơi chứa cac_quy_dinh_..., tomtat...
COLL_PREFIX   = "benchmark_"             # Qdrant collection => benchmark_800_400
CONFIGS: List[Tuple[int,int]] = [
    (1000,100), (800,400), (800,200), (400,200)
]
EVAL_FILE     = Path("tests/eval_questions.jsonl")
EMB_MODEL_ID  = cfg.LOCAL_EMBEDDING_MODEL_NAME
QDRANT_URL    = cfg.QDRANT_URL
QDRANT_KEY    = cfg.QDRANT_API_KEY

# ---------- Khởi tạo ----------
embedder = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    model_kwargs={"device": "cuda"})      # or "cpu"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, prefer_grpc=False)

# ---------- Helper ----------
def ingest_collection(col_name:str, chunk_size:int, overlap:int):
    """Tách file, embed, upsert vào Qdrant collection col_name"""
    # tạo collection nếu chưa có
    try:
        client.get_collection(col_name)
        print(f"[INFO] Collection {col_name} đã tồn tại – bỏ qua ingest.")
        return
    except Exception:
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    total_chunks = 0
    for fpath in DATA_DIR.glob("*.md"):
        loader = UnstructuredMarkdownLoader(str(fpath))
        docs   = loader.load()
        chunks = splitter.split_documents(docs)
        vecs   = [embedder.embed_query(c.page_content) for c in chunks]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[i],
                payload={"content": chunks[i].page_content}
            ) for i in range(len(chunks))
        ]
        client.upsert(col_name, points)
        total_chunks += len(points)
        print(f"[OK] {fpath.name}: {len(points)} chunks")
    print(f"[DONE] Ingest {total_chunks} chunks vào {col_name}")

def precision_at_3(col_name:str) -> float:
    """Tính Precision@3 với collection col_name"""
    with open(EVAL_FILE, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]

    hit = 0
    for s in samples:
        qvec = embedder.embed_query(s["question"])
        res  = client.search(
            collection_name=col_name,
            query_vector=qvec,
            limit=3,
            with_payload=True)
        pred_ids = {h.id for h in res}
        gold_ids = set(s["gold_ids"])
        hit += len(pred_ids & gold_ids)
    return hit / (3*len(samples))

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Ingest collections")
    parser.add_argument("--eval",  action="store_true", help="Evaluate P@3")
    args = parser.parse_args()

    if args.build:
        for cs, ov in CONFIGS:
            ingest_collection(f"{COLL_PREFIX}{cs}_{ov}", cs, ov)

    if args.eval:
        print("\n=== Precision@3 ===")
        for cs, ov in CONFIGS:
            col = f"{COLL_PREFIX}{cs}_{ov}"
            p   = precision_at_3(col)
            print(f"{cs}-{ov}: {p:.2f}")
