# app/routes/chat_ws.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.models import ChatHistory, Conversation
from app.utils.security import get_current_user
from app.pipelines.rag_pipeline import RAGPipelineLoader
from typing import List

router = APIRouter()
rag_pipeline_loader = RAGPipelineLoader()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # Đây thực hiện websocket.accept()
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    # 1. Lấy token
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Xác thực
    try:
        current_user = get_current_user(token=token, db=db)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Accept và thêm connection
    await manager.connect(websocket)

    # 4. Xác định hoặc tạo conversation
    conv_id = websocket.query_params.get("conversation_id")
    if conv_id:
        conv = db.query(Conversation).filter_by(
            id=int(conv_id), user_id=current_user.id
        ).first()
        if not conv:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    else:
        conv = Conversation(user_id=current_user.id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        #  gửi conversation_id sau khi đã accept
        await manager.send_personal_message(f"conversation_id:{conv.id}", websocket)

    try:
        while True:
            # 5. Nhận câu hỏi
            data = await websocket.receive_text()

            # 6. Query RAG
            result = rag_pipeline_loader.rag(source="qdrant", question=data)
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])

            # 7. Lưu vào DB
            chat = ChatHistory(
                user_id=current_user.id,
                conversation_id=conv.id,
                question=data,
                answer=answer,
                rag_context="\n\n".join(d.page_content for d in source_documents)
            )
            db.add(chat)
            db.commit()

            # 8. Gửi lại kết quả
            await manager.send_personal_message(
                f"You asked: {data}\nAssistant: {answer}",
                websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
