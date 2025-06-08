from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.config.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="student")  # 'student' hoặc 'admin'

    chats = relationship("ChatHistory", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rag_context = Column(Text, nullable=True)  # Lưu trữ đoạn chat của RAG

    user = relationship("User", back_populates="chats")

class Regulation(Base):
    __tablename__ = "regulations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    qdrant_collection = Column(String, nullable=True)
    qdrant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Ví dụ về model cho các loại tài liệu tải lên khác (nếu cần)
# class UploadedDocument(Base):
#     __tablename__ = "uploaded_documents"
#
#     id = Column(Integer, primary_key=True, index=True)
#     admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     file_name = Column(String, nullable=False)
#     file_path = Column(String, nullable=False)
#     qdrant_collection = Column(String, nullable=True)
#     qdrant_id = Column(String, nullable=True)
#     upload_timestamp = Column(DateTime, default=func.now())
#
#     admin = relationship("User")

# Model cho Câu hỏi thường gặp (FAQ)
class FAQ(Base):
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    qdrant_collection = Column(String, nullable=True)
    qdrant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Model cho Vấn đề liên quan
class RelatedIssue(Base):
    __tablename__ = "related_issues"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    guidance_link = Column(String, nullable=False)
    qdrant_collection = Column(String, nullable=True)
    qdrant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class Conversation(Base):
    __tablename__ = "conversations"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String(255), nullable=True)             # bạn có thể cho user đặt tên
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="conversations")
    chats = relationship("ChatHistory", back_populates="conversation", cascade="all, delete")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)  # — THÊM FK
    timestamp       = Column(DateTime, default=func.now())
    question        = Column(Text, nullable=False)
    answer          = Column(Text, nullable=False)
    rag_context     = Column(Text, nullable=True)

    user = relationship("User", back_populates="chats")
    conversation = relationship("Conversation", back_populates="chats")  # — THÊM quan hệ