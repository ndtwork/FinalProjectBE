# app/routes/auth.py

import os
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from .. import models, schemas
from ..config.database import get_db
from ..utils.security import verify_password, create_access_token, hash_password
from app.config.config import ACCESS_TOKEN_EXPIRE_MINUTES

# Đọc credential Admin từ .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", f"{ADMIN_USERNAME}@example.com")
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise RuntimeError("Bạn cần set ADMIN_USERNAME và ADMIN_PASSWORD trong .env")

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Chỉ tạo user role="student"
    exists = db.query(models.User).filter(
        (models.User.username == user.username) |
        (models.User.email    == user.email)
    ).first()
    if exists:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    db_user = models.User(
        username        = user.username,
        email           = user.email,
        hashed_password = hash_password(user.password),
        role            = "student"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=schemas.Token)
def login_user(user: schemas.LoginRequest, db: Session = Depends(get_db)):
    # 1. Nếu login bằng ENV-admin credentials
    if user.username == ADMIN_USERNAME and user.password == ADMIN_PASSWORD:
        # Kiểm tra xem admin đã có trong DB chưa
        db_admin = db.query(models.User).filter_by(username=ADMIN_USERNAME).first()
        if not db_admin:
            # Tạo record admin đầu tiên trong DB
            db_admin = models.User(
                username        = ADMIN_USERNAME,
                email           = ADMIN_EMAIL,
                hashed_password = hash_password(ADMIN_PASSWORD),
                role            = "admin"
            )
            db.add(db_admin)
            db.commit()
            db.refresh(db_admin)
        # Issue token dựa trên record DB mới tạo
        access_token = create_access_token(
            subject        = db_admin.username,
            role           = db_admin.role,
            expires_delta  = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": access_token, "token_type": "bearer"}

    # 2. Ngược lại: lookup trong DB
    db_user = db.query(models.User).filter_by(username=user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token(
        subject        = db_user.username,
        role           = db_user.role,
        expires_delta  = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}
