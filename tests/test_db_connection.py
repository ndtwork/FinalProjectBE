import psycopg2
import os
from dotenv import load_dotenv
print("Bắt đầu kiểm tra kết nối...")
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")  # Mặc định là 5432

try:
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    print("Kết nối PostgreSQL thành công!")

    # Bạn có thể thực hiện một truy vấn đơn giản để kiểm tra
    cur.execute("SELECT version();")
    db_version = cur.fetchone()
    print(f"Phiên bản PostgreSQL: {db_version}")

    cur.close()
    conn.close()

except psycopg2.Error as e:
    print(f"Lỗi kết nối PostgreSQL: {e}")
except Exception as e:
    print(f"Đã xảy ra lỗi khác: {e}")