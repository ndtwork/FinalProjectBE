# app/routes/chat_history.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.config.database import get_db
from app.models import ChatHistory, Conversation
from app.schemas import ChatHistoryOut, ConversationOut
from app.utils.security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])

# — Bulk delete chats by IDs
@router.delete("/history")
def bulk_delete(
    ids: List[int] = Query(..., description="List of chat IDs to delete"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    q = db.query(ChatHistory).filter(
        ChatHistory.user_id == current_user.id,
        ChatHistory.id.in_(ids)
    )
    count = q.count()
    if count == 0:
        raise HTTPException(404, "No chat records found to delete")
    q.delete(synchronize_session=False)
    db.commit()
    return {"message": f"Deleted {count} chat records."}

# — List conversations
@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    convs = (
        db.query(Conversation)
          .filter(Conversation.user_id == current_user.id)
          .order_by(Conversation.created_at.desc())
          .all()
    )
    return convs

# — Create a conversation (optional, since WS auto-creates)
@router.post("/conversations", response_model=ConversationOut)
def create_conversation(
    title: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    conv = Conversation(user_id=current_user.id, title=title)
    db.add(conv); db.commit(); db.refresh(conv)
    return conv

# — Get messages in a conversation
@router.get("/conversations/{conv_id}/messages", response_model=List[ChatHistoryOut])
def get_conversation_messages(
    conv_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    conv = db.query(Conversation).filter_by(id=conv_id, user_id=current_user.id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msgs = (
        db.query(ChatHistory)
          .filter(ChatHistory.conversation_id == conv_id)
          .order_by(ChatHistory.timestamp.asc())
          .offset(skip).limit(limit)
          .all()
    )
    return msgs

# — Delete a full conversation
@router.delete("/conversations/{conv_id}")
def delete_conversation(
    conv_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    conv = db.query(Conversation).filter_by(id=conv_id, user_id=current_user.id).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()
    return {"message": f"Deleted conversation {conv_id} and its messages."}
