# app/routes/chat_ws.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from app.models import ChatHistory, Conversation   # — THÊM Conversation
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.models import ChatHistory
from app.utils.security import get_current_user
from app.pipelines.rag_pipeline import RAGPipelineLoader
from typing import List

router = APIRouter()
rag_pipeline_loader = RAGPipelineLoader()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
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
    # 1. Lấy token từ query param
    token = websocket.query_params.get("token")
    if not token:
        # Đóng kết nối nếu không có token
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Xác thực token và lấy current_user
    try:
        current_user = get_current_user(token=token, db=db)
    except HTTPException:
        # Nếu token không hợp lệ hoặc user không tìm thấy
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Kết nối WebSocket
    await manager.connect(websocket)

    try:
        while True:
            # 4. Nhận câu hỏi từ client
            data = await websocket.receive_text()

            # 5. Xử lý RAG query
            result = rag_pipeline_loader.rag(source="qdrant", question=data)
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])

            # 6. Lưu vào ChatHistory
            chat = ChatHistory(
                user_id=current_user.id,
                question=data,
                answer=answer,
                rag_context="\n\n".join(d.page_content for d in source_documents)
            )
            db.add(chat)
            db.commit()

            # 7. Gửi lại kết quả cho client
            await manager.send_personal_message(
                f"You asked: {data}\nAssistant: {answer}",
                websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
