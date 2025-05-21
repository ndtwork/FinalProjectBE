# app/config/config.py
import os
from datetime import timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")  # Thay bằng khóa bí mật thực tế của bạn
# Nếu không có biến môi trường, sử dụng giá trị mặc định YOUR_SECRET_KEY, còn SECRET_KEY là biến môi trường trong file .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # thời gian hết hạn access token