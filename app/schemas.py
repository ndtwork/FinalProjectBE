from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    role: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Có thể bạn sẽ cần thêm các Pydantic model cho Regulation, FAQ, và RelatedIssue sau này
# để định nghĩa cấu trúc dữ liệu cho request và response liên quan đến chúng

from datetime import datetime
from typing import Optional
class ChatHistoryOut(BaseModel):
    id: int
    timestamp: datetime
    question: str
    answer: str
    rag_context: Optional[str]
    class Config:
        orm_mode = True

