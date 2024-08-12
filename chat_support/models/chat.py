from pydantic import BaseModel
from pydantic.types import UUID4

# This model is intended to represent the response of a chat system.
class ChatResponse(BaseModel):
    message: str
    id: UUID4

#This model is intended to represent the request of a chat system.
class ChatRequest(BaseModel):
    question: str
    stream: bool = False