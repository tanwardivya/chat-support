import time
import os
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from chat_support.routers import chat
from chat_support.utils.dependency import VectorStore   
@asynccontextmanager
async def lifespan(app: FastAPI):
    VectorStore.load_index()
    VectorStore.initialize_bedrock_client()
    VectorStore.embedding_client()
    VectorStore.create_index()
    VectorStore.initialize_llm()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def root():
    return {"message": "Chat Support for FAQs"}