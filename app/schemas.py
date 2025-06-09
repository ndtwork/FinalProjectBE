# app/schemas.py

from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    role: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# --- Conversation schema để list/create/xóa conversations ---
class ConversationOut(BaseModel):
    id: int
    title: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True

# --- ChatHistory schema để trả về messages trong conversation ---
class ChatHistoryOut(BaseModel):
    id: int
    conversation_id: int
    timestamp: datetime
    question: str
    answer: str
    rag_context: Optional[str]

    class Config:
        orm_mode = True
