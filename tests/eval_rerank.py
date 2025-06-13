# tests/eval_rerank.py
"""
Đánh giá Precision@3 trước & sau rerank.
Không đụng vào code production: chỉ import RAGPipelineLoader.
"""

import json
import time
import numpy as np
import sys
from pathlib import Path

# --- đảm bảo project_root nằm trong PYTHONPATH ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- import lớp pipeline đã viết ---
from app.pipelines.rag_pipeline import RAGPipelineLoader

# Nếu muốn override collection khác với config mặc định, bạn có thể:
#   1. Đặt biến môi trường trước khi chạy: export QDRANT_COLLECTION_NAME="tên_collection"
#   2. Hoặc gán thủ công sau khi init loader:
#      loader.collection_name = "tên_collection"
COLLECTION = "1000v100"

# Khởi pipeline
loader = RAGPipelineLoader()
# Khởi retriever với signature gốc: load_retriever(self, retriever_name, embeddings)
loader.load_retriever("qdrant", loader.embeddings)
loader.source = "qdrant"

# Nếu muốn override ngay trong script:
loader.collection_name = COLLECTION

# Tham số đánh giá
K, N = 11, 3  # K = tổng số chunk lấy trước rerank, N = số chunk giữ lại sau rerank

def eval_one(question, gold_ids):
    # 1. Embed câu hỏi
    qvec = loader.embeddings.embed_query(question)
    # 2. Lấy chunk theo từng document_type
    regs   = loader._retrieve_chunks(qvec, "Regulation",    3)
    faqs   = loader._retrieve_chunks(qvec, "FAQ",           3)
    issues = loader._retrieve_chunks(qvec, "RelatedIssue",  3)
    raw    = regs + faqs + issues  # tổng khoảng 9–11 chunk

    # Precision@N trước rerank
    pre_ids = [
        d.metadata.get("id")
        for d in sorted(raw, key=lambda d: -d.metadata["score"])[:N]
    ]
    p_pre = len(set(pre_ids) & set(gold_ids)) / N

    # Rerank và đo latency
    t0 = time.time()
    topN = loader._rerank(question, raw, top_n=N)
    t_rerank = time.time() - t0
    post_ids = [d.metadata.get("id") for d in topN]
    p_post = len(set(post_ids) & set(gold_ids)) / N

    return p_pre, p_post, t_rerank

def main():
    # Đường dẫn file chứa câu hỏi & gold_ids
    data_path = Path(__file__).with_name("eval_questions.jsonl")
    prec_pre, prec_post, times = [], [], []

    # Đọc lần lượt từng dòng JSONL
    for line in data_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        p_pre, p_post, t = eval_one(sample["question"], sample["gold_ids"])
        prec_pre.append(p_pre)
        prec_post.append(p_post)
        times.append(t)

    # In kết quả trung bình
    print(f"Precision@{N} trước rerank : {np.mean(prec_pre):.2f}")
    print(f"Precision@{N} sau   rerank : {np.mean(prec_post):.2f}")
    print(f"Latency rerank trung bình : {1000 * np.mean(times):.1f} ms")

if __name__ == "__main__":
    main()