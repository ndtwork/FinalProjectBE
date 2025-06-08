# tests/eval_rerank.py
"""
Đánh giá Precision@3 trước & sau rerank.
Không đụng vào code production: chỉ import RAGPipelineLoader.
"""

import json, time, numpy as np, sys
from pathlib import Path

# --- đảm bảo project_root nằm trong PYTHONPATH ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- import lớp pipeline đã viết ---
from app.pipelines.rag_pipeline import RAGPipelineLoader   # <-- lấy code gốc

loader = RAGPipelineLoader()      # khởi tạo
# ► khởi tạo retriever & qdrant_client
loader.load_retriever("qdrant", loader.embeddings)
loader.source = "qdrant"

K, N = 11, 3                      # lấy rộng 11 ➜ rerank giữ 3

def eval_one(question, gold_ids):
    """Trả về precision trước & sau rerank + thời gian rerank (s)"""
    # 1. embed & search K chunk
    qvec  = loader.embeddings.embed_query(question)
    regs  = loader._retrieve_chunks(qvec, "Regulation", 3)
    faqs  = loader._retrieve_chunks(qvec, "FAQ", 3)
    issues= loader._retrieve_chunks(qvec, "RelatedIssue", 3)
    raw   = regs + faqs + issues                 # ~9–11 chunk

    # ---- Precision@3 trước rerank ----
    pre_ids = [d.metadata.get("id") for d in sorted(raw, key=lambda x: -x.metadata["score"])[:N]]
    p_pre   = len(set(pre_ids) & set(gold_ids)) / N

    # ---- Rerank giữ 3 ----
    t0      = time.time()
    top3    = loader._rerank(question, raw, top_n=N)
    rerank_time = time.time() - t0
    post_ids = [d.metadata.get("id") for d in top3]
    p_post   = len(set(post_ids) & set(gold_ids)) / N
    return p_pre, p_post, rerank_time


def main():
    prec_pre, prec_post, times = [], [], []
    data_path = Path(__file__).with_name("eval_questions.jsonl")
    for line in data_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        p_pre, p_post, t = eval_one(sample["question"], sample["gold_ids"])
        prec_pre.append(p_pre); prec_post.append(p_post); times.append(t)

    print(f"Precision@3  trước rerank : {np.mean(prec_pre):.2f}")
    print(f"Precision@3  sau   rerank : {np.mean(prec_post):.2f}")
    print(f"Latency rerank trung bình : {1000*np.mean(times):.1f} ms")


if __name__ == "__main__":
    main()
