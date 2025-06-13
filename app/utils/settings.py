# app/utils/settings.py
import json
from pathlib import Path

# Lưu file trong thư mục config (bên cạnh rag_config.py)
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "active_collection.json"

def get_active_collection() -> str | None:
    """Trả về tên collection hiện tại, hoặc None nếu chưa set."""
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data.get("active_collection")
    except FileNotFoundError:
        return None

def set_active_collection(name: str):
    """Ghi đè active_collection vào file JSON."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps({"active_collection": name}), encoding="utf-8")
