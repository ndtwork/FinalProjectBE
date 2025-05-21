from fastapi import FastAPI
from . import models
from app.config.database import engine
from app.routes import auth, chat_ws  # Thêm import chat_ws

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router)  # Include the authentication router
app.include_router(chat_ws.router)  # Include the WebSocket router

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}