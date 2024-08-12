# filename: bedrock_embedder.py  
import streamlit as st  
import numpy as np  
import os
import errno
import boto3
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import BedrockEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from loguru import logger


def create_s3_client():
    s3_client = boto3.client('s3')
    return s3_client

def create_bedrock_client():
    bedrock_client = boto3.client(service_name = "bedrock-runtime")
    return bedrock_client

BEDROCK_CLIENT = create_bedrock_client()
def embedding_client(model_id: str = "amazon.titan-embed-text-v1") -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id = model_id, client = BEDROCK_CLIENT)


S3_CLIENT= create_s3_client()

EMBEDDINGS = embedding_client(model_id='amazon.titan-embed-text-v1')
BUCKET_NAME = "headstarter-demo-bedrock-chat-pdf"
FOLDER_PATH = "/tmp/"
def load_index():
    S3_CLIENT.download_file(Bucket = BUCKET_NAME, Key = "my_faiss.faiss", Filename = f"{FOLDER_PATH}my_faiss.faiss")
    S3_CLIENT.download_file(Bucket = BUCKET_NAME, Key = "my_faiss.pkl", Filename = f"{FOLDER_PATH}my_faiss.pkl")
    logger.info("Index loaded successfully.")


def rag_search(prompt: str, index_path) -> list:  
    allow_dangerous = True  
    index_file_path = os.path.join(index_path, "my_faiss.faiss")
    if not os.path.exists(index_file_path):
        return [f"Index file {index_file_path} does not exist. Please follow these steps to create it: \n"
                "1. Upload the text or PDF files you want to index. \n"
                "2. Look for the 'Index' button at the bottom of the sidebar. \n"
                "3. Click the 'Index' button to index the files. \n"
                "The indexed files will be saved in the created folder and will be used as your local index."]
    db = FAISS.load_local(index_name="my_faiss", folder_path = FOLDER_PATH, embeddings = EMBEDDINGS, allow_dangerous_deserialization=allow_dangerous)
    docs = db.similarity_search(prompt, k=5)  
    return docs

def search_index(prompt: str, index_path: str = FOLDER_PATH) -> list:  
    if prompt:  
        matching_docs = rag_search(prompt, index_path) 
        return matching_docs
    else:  
        return []


