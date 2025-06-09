# scripts/reset_db.py
from app.config.database import Base, engine

def reset_database():
    # Xóa tất cả bảng hiện có
    Base.metadata.drop_all(bind=engine)
    # Tạo lại theo models mới
    Base.metadata.create_all(bind=engine)
    print("✅ Đã reset database schema")

if __name__ == "__main__":
    reset_database()
