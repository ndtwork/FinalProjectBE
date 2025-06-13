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

    def connect(self, websocket: WebSocket):
        # Chỉ track connection, không gọi accept() ở đây
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    # 1. Lấy token từ query string
    token = websocket.query_params.get("token")
    if not token:
        # Không có token → đóng kết nối
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Xác thực JWT và fetch user
    try:
        current_user = get_current_user(token=token, db=db)
    except HTTPException:
        # Token không hợp lệ → đóng kết nối
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Accept WebSocket handshake và track connection
    await websocket.accept()
    manager.connect(websocket)

    # 4. Xác định conversation
    conv_id = websocket.query_params.get("conversation_id")
    if conv_id:
        conv = db.query(Conversation).filter_by(
            id=int(conv_id), user_id=current_user.id
        ).first()
        if not conv:
            # conv không tồn tại hoặc không phải của user → đóng
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            manager.disconnect(websocket)
            return
    else:
        # Tạo mới conversation
        conv = Conversation(user_id=current_user.id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        # Gửi trả về cho client conversation_id
        await manager.send_personal_message(f"conversation_id:{conv.id}", websocket)

    # 5. Vòng lặp nhận/sent messages
    try:
        while True:
            data = await websocket.receive_text()

            # 5.1. Chạy RAG pipeline
            result = rag_pipeline_loader.rag(source="qdrant", question=data)
            answer = result.get("result", "")
            docs   = result.get("source_documents", [])

            # 5.2. Lưu vào DB
            chat = ChatHistory(
                user_id=current_user.id,
                conversation_id=conv.id,
                question=data,
                answer=answer,
                rag_context="\n\n".join(d.page_content for d in docs)
            )
            db.add(chat)
            db.commit()

            # 5.3. Gửi kết quả về client
            await manager.send_personal_message(
                f"You asked: {data}\nAssistant: {answer}",
                websocket
            )

    except WebSocketDisconnect:
        # Client đóng kết nối
        manager.disconnect(websocket)
