from fastapi import FastAPI
from . import models
from app.config.database import engine
from app.routes import auth, chat_ws  # Thêm import chat_ws
from app.routes.rag_admin import router as rag_admin_router
from app.routes.chat_history import router as chat_history_router
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # hoặc ["*"] để test nhanh
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth.router)  # Include the authentication router
app.include_router(chat_ws.router)  # Include the WebSocket router
app.include_router(rag_admin_router)
app.include_router(chat_history_router)

# … include_router các router đã có …

from fastapi import FastAPI
from app.config.database import get_db
from sqlalchemy.orm import Session
import os

from app.utils.security import hash_password
from app.models import User

@app.on_event("startup")
def seed_admin():
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
    ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", f"{ADMIN_USERNAME}@example.com")

    db: Session = next(get_db())
    # Nếu chưa có trong DB, tạo luôn
    if not db.query(User).filter_by(username=ADMIN_USERNAME).first():
        admin = User(
            username        = ADMIN_USERNAME,
            email           = ADMIN_EMAIL,
            hashed_password = hash_password(ADMIN_PASSWORD),
            role            = "admin"
        )
        db.add(admin)
        db.commit()
    db.close()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}