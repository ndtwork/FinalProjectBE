import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import re
load_dotenv()

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

project_root = Path(__file__).resolve().parents[1]
EVAL_FILE = project_root / "tests" / "eval_questions.jsonl"
samples = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines()]

COLLECTIONS = {
    "1000v100":  (5, 3, 3),
    "800v400":   (5, 3, 3),
    "800v200":   (5, 3, 3),
    "400v200":   (5, 3, 3),
}

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)

def embed_query(text):
    return embedder.embed_query(text)

def search_qdrant(collection_name, query_vec, filter_type, top_k):
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k,
        query_filter={
            "must": [
                {"key": "metadata.document_type", "match": {"value": filter_type}}
            ]
        }
    )
    return hits

def normalize(text):
    # Chuẩn hóa đơn giản để so khớp "content" cho bền vững
    return re.sub(r'\s+', ' ', text.strip().lower())

def precision_for_collection(col_name: str, reg_k, faq_k, iss_k):
    hit_pre = 0
    for s in samples:
        q = s["question"]
        # -- Lấy set nội dung chuẩn hóa --
        gold_contents = set([normalize(x) for x in s.get("gold_contents", [])])

        qvec = embed_query(q)
        regs = search_qdrant(col_name, qvec, "Regulation", reg_k)
        faqs = search_qdrant(col_name, qvec, "FAQ", faq_k)
        issues = search_qdrant(col_name, qvec, "RelatedIssue", iss_k)
        raw = list(regs) + list(faqs) + list(issues)

        sorted_chunks = sorted(raw, key=lambda x: -x.score)
        top3 = sorted_chunks[:3]
        # -- Lấy nội dung trả về --
        pred_contents = set([normalize(chunk.payload["content"]) for chunk in top3])

        hit_pre += len(pred_contents & gold_contents)
        # Nếu muốn debug:
        # print(f"Q: {q}")
        # print("PRED:", [chunk.payload["content"][:60] for chunk in top3])
        # print("GOLD:", gold_contents)
        # print("==>", pred_contents & gold_contents)

    total = 3 * len(samples)
    return hit_pre / total if total > 0 else 0.0

def main():
    print("=== Precision@3 Qdrant độc lập ===")
    print("Collection    Precision@3")
    for col in COLLECTIONS:
        reg_k, faq_k, iss_k = COLLECTIONS[col]
        p_pre = precision_for_collection(col, reg_k, faq_k, iss_k)
        print(f"{col:<12} {p_pre:6.2%}")

if __name__ == "__main__":
    main()
