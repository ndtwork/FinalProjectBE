import json
from pathlib import Path
from qdrant_client import QdrantClient

# Cấu hình Qdrant
QDRANT_URL = "https://43965d09-2062-4e87-9d29-fed11a204a3c.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bdx9pR6wJycDyXQu6f4Al7xAU3wSiozwLjaDV0FyiLY"
COLLECTION = "400v200"  # hoặc chọn collection nào bạn muốn

# Load danh sách câu hỏi từ file cũ
EVAL_FILE = Path("eval_questions.jsonl")
samples = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()]

# Kết nối Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_chunk_content(chunk_id):
    """Trả về content của chunk theo id trong Qdrant"""
    res = client.retrieve(
        collection_name=COLLECTION,
        ids=[chunk_id]
    )
    if not res:
        print(f"[WARN] Không tìm thấy chunk id: {chunk_id}")
        return None
    return res[0].payload.get("content", "")

# Build lại samples với gold_contents
new_samples = []
for s in samples:
    gold_ids = s.get("gold_ids", [])
    gold_contents = []
    for gid in gold_ids:
        content = get_chunk_content(gid)
        if content:
            gold_contents.append(content)
    s["gold_contents"] = gold_contents
    new_samples.append(s)

# Ghi ra file mới
output_file = Path("eval_questions_with_contents.jsonl")
with output_file.open("w", encoding="utf-8") as f:
    for s in new_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("Đã sinh file eval_questions_with_contents.jsonl thành công!")
