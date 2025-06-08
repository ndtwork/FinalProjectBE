# app/utils/security.py

from passlib.context import CryptContext
from datetime import timedelta, datetime
from jose import jwt, JWTError                                     # — THÊM JWTError
from fastapi import Depends, HTTPException, status                # — THÊM Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer                  # — THÊM OAuth2PasswordBearer

from sqlalchemy.orm import Session                                  # — THÊM Session
from app.config.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.config.database import get_db                              # — THÊM get_db
from app.models import User                                         # — THÊM User model

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# — THÊM: OAuth2 scheme để FastAPI tự động đọc header Authorization: Bearer <token>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(subject: str, role: str, expires_delta: timedelta | None = None):
    """
    Tạo JWT với payload:
      - sub: subject (thường là email hoặc username)
      - role: quyền (e.g. "admin" / "student")
      - exp: thời điểm hết hạn
    """
    to_encode = {"sub": subject, "role": role}
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# — THÊM: Dependency để lấy user hiện tại từ JWT
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    1. Lấy token từ header Authorization: Bearer <token>
    2. Giải mã JWT, kiểm tra 'sub' (username/email) và 'exp'
    3. Truy vấn DB để fetch user
    4. Nếu không hợp lệ hoặc user không tồn tại → 401 Unauthorized
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# — THÊM: Dependency chuyên biệt cho admin
def get_current_admin_user(current_user = Depends(get_current_user)):
    """
    Kiểm tra field 'role' của user phải là 'admin',
    nếu không → 403 Forbidden
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user
