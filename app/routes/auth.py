# app/routes/auth.py
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import timedelta
from .. import models, schemas
from ..config.database import get_db
from ..utils.security import verify_password, create_access_token, hash_password
from app.config.config import ACCESS_TOKEN_EXPIRE_MINUTES

#
# Đọc tài khoản admin từ biến môi trường; bắt buộc có trong .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise RuntimeError("Bạn cần set ADMIN_USERNAME và ADMIN_PASSWORD trong .env")


router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Chỉ tạo user role="student"
    db_user = db.query(models.User).filter(
        (models.User.username == user.username) |
        (models.User.email == user.email)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    hashed_password = hash_password(user.password)
    db_user = models.User(
    username = user.username,
    email = user.email,
    hashed_password = hashed_password,
    role = "student"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=schemas.Token)
def login_user(user: schemas.LoginRequest, db: Session = Depends(get_db)):
    # 1. Nếu trùng với ADMIN_USERNAME, ADMIN_PASSWORD thì tạo token admin
    if user.username == ADMIN_USERNAME and user.password == ADMIN_PASSWORD:
        token = create_access_token(
            subject = ADMIN_USERNAME,
            role = "admin",
            expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": token, "token_type": "bearer"}

    # 2. Ngược lại lookup từ DB, chỉ login được user có role student (hoặc admin tự seed trong DB)
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token(
        subject = db_user.username,
        role = db_user.role,
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}