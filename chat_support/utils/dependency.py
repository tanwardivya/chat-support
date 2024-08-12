import os
from loguru import logger
import boto3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

BUCKET_NAME = os.getenv("BUCKET_NAME")
FOLDER_PATH = "/tmp/"
def create_s3_client():
    s3_client = boto3.client('s3')
    return s3_client


def create_bedrock_client():
    bedrock_client = boto3.client(service_name = "bedrock-runtime")
    return bedrock_client

class VectorStore:
    s3_client = None
    faiss_index = None
    bedrock_client = None
    bedrock_embeddings = None
    bedrock_llm = None
    @staticmethod
    def load_index():
        VectorStore.s3_client = create_s3_client()
        VectorStore.s3_client.download_file(Bucket = BUCKET_NAME, Key = "my_faiss.faiss", Filename = f"{FOLDER_PATH}my_faiss.faiss")
        VectorStore.s3_client.download_file(Bucket = BUCKET_NAME, Key = "my_faiss.pkl", Filename = f"{FOLDER_PATH}my_faiss.pkl")
        logger.info("Index loaded successfully.")

    @staticmethod
    def initialize_bedrock_client():
        VectorStore.bedrock_client = create_bedrock_client()
        logger.info("Bedrock client initialized successfully.")
    @staticmethod
    def embedding_client():
        VectorStore.bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = VectorStore.bedrock_client)
        logger.info("Bedrock embeddings client initialized successfully.")

    @staticmethod
    def create_index():
        if VectorStore.bedrock_embeddings:
            VectorStore.faiss_index = FAISS.load_local(index_name="my_faiss", folder_path = FOLDER_PATH,embeddings = VectorStore.bedrock_embeddings, allow_dangerous_deserialization=True)
            logger.info("Index created successfully.")
        else:
            logger.error("Bedrock embeddings client not initialized.")
    
    @staticmethod
    def initialize_llm():
        VectorStore.bedrock_llm = Bedrock(model_id= "anthropic.claude-v2:1", client = VectorStore.bedrock_client, model_kwargs={'max_tokens_to_sample':512})
        logger.info("LLM initialized successfully.")

            