import os
import uuid

from fastapi import APIRouter, Body, HTTPException, Request, status, Response
from loguru import logger

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from chat_support.models.chat import ChatResponse, ChatRequest
from chat_support.utils.dependency import VectorStore

router = APIRouter()

@router.post("/chat", tags=["chats"], response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    
    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
    llm=VectorStore.bedrock_llm,
    chain_type="stuff",
    retriever=VectorStore.faiss_index.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ) if VectorStore.faiss_index else None,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":chat_request.question})

    return ChatResponse(message=answer['result'], id=uuid.uuid4())
