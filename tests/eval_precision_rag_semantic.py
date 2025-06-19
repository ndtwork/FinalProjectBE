import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder  # NEW

# --------- Config ----------
project_root = Path(__file__).resolve().parents[1]
EVAL_FILE = project_root / "tests" / "generated_eval_from_800v400.jsonl"
samples = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines()]

COLLECTIONS = {
    "1000v100":  (5, 3, 3),
    "800v400":   (5, 3, 3),
    "800v200":   (5, 3, 3),
    "400v200":   (5, 3, 3),
    "200v100":   (5, 3, 3),
    "100v50":    (5, 3, 3),
    "20v10":     (5, 3, 3),
}

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)

# NEW: Load cross-encoder model for re-ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

def embed_query(text):
    return embedder.embed_query(text)

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

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

def precision_for_collection(col_name: str, reg_k, faq_k, iss_k, threshold=0.83):
    hit_pre = 0
    for s in samples:
        q = s["question"]
        gold_contents = s.get("gold_contents", [])
        if not gold_contents:
            continue  # Bỏ qua nếu không có đáp án vàng

        gold_vecs = [embed_query(gc) for gc in gold_contents]
        qvec = embed_query(q)

        # Step 1: Get initial retrieved results from Qdrant
        regs = search_qdrant(col_name, qvec, "Regulation", reg_k)
        faqs = search_qdrant(col_name, qvec, "FAQ", faq_k)
        issues = search_qdrant(col_name, qvec, "RelatedIssue", iss_k)

        # Step 2: Re-rank all retrieved results using cross-encoder
        all_chunks = regs + faqs + issues
        rerank_inputs = [(q, chunk.payload["content"]) for chunk in all_chunks]
        rerank_scores = reranker.predict(rerank_inputs)
        scored_chunks = list(zip(all_chunks, rerank_scores))
        sorted_chunks = sorted(scored_chunks, key=lambda x: -x[1])
        top3 = [chunk for chunk, _ in sorted_chunks[:3]]

        # Step 3: Embed top3 predicted chunks and compare to gold
        pred_vecs = [embed_query(chunk.payload["content"]) for chunk in top3]

        match = False
        for gold in gold_vecs:
            for pred in pred_vecs:
                if cosine_sim(gold, pred) > threshold:
                    match = True
                    break
            if match:
                break

        if match:
            hit_pre += 1

        # Optional debug print
        # print(f"Q: {q}")
        # print("PRED:", [chunk.payload["content"][:80] for chunk in top3])
        # print("GOLD:", gold_contents)
        # print("==>", match)
        # print("")

    total = len([s for s in samples if s.get("gold_contents")])
    return hit_pre / total if total > 0 else 0.0

def main():
    print("=== Precision@3 (Semantic, re-rank by CrossEncoder) ===")
    print("Collection    Precision@3")
    for col in COLLECTIONS:
        reg_k, faq_k, iss_k = COLLECTIONS[col]
        p_pre = precision_for_collection(col, reg_k, faq_k, iss_k, threshold=0.7)
        print(f"{col:<12} {p_pre:6.2%}")

if __name__ == "__main__":
    main()
