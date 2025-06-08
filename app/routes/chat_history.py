from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.config.database import get_db
from app.models import ChatHistory
from app.schemas import ChatHistoryOut
from app.utils.security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])

@router.get(
    "/history",
    response_model=List[ChatHistoryOut],
    summary="Lấy lịch sử chat của user hiện tại"
)
def get_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    entries = (
        db.query(ChatHistory)
          .filter(ChatHistory.user_id == current_user.id)
          .order_by(ChatHistory.timestamp.desc())
          .offset(skip)
          .limit(limit)
          .all()
    )
    return entries

@router.delete(
    "/history/{chat_id}",
    summary="Xóa 1 bản ghi chat của user"
)
def delete_history(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    entry = db.query(ChatHistory).filter_by(id=chat_id, user_id=current_user.id).first()
    if not entry:
        raise HTTPException(404, detail="Chat record not found")
    db.delete(entry)
    db.commit()
    return {"message": f"Deleted chat {chat_id}"}
