from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.pipelines.rag_pipeline import RAGPipelineLoader  # Import RAGPipelineLoader
from typing import List

router = APIRouter()
rag_pipeline_loader = RAGPipelineLoader()  # Khởi tạo RAGPipelineLoader

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
    token = websocket.query_params.get("token")
    print(f"Received token: {token}")  # In để kiểm tra token

    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Sử dụng RAGPipelineLoader để xử lý câu hỏi
            result = rag_pipeline_loader.rag(source="qdrant", question=data)
            answer = result['result']
            await manager.send_personal_message(f"You asked: {data}\nAssistant: {answer}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)