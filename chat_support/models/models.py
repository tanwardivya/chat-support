from typing import Dict, List, Union
from config import config

from langchain_aws import ChatBedrockConverse
from loguru import logger
import boto3
class ChatModel:
    def __init__(self, model_name: str, model_kwargs: Dict):
        self.model_config = config["models"][model_name]
        self.model_id = self.model_config["model_id"]
        self.model_kwargs = model_kwargs
        bedrock_client = boto3.client(service_name = "bedrock-runtime", region_name='us-east-1')
        logger.info(f"Initializing {model_name} model with ID: {self.model_id}")
        self.llm = ChatBedrockConverse(model=self.model_id, max_tokens=self.model_kwargs["max_tokens"], temperature=self.model_kwargs["temperature"], top_p=self.model_kwargs["top_p"], additional_model_request_fields={"top_k": self.model_kwargs["top_k"]},
                   client=bedrock_client)