from pydantic import BaseModel
from datetime import datetime

class MessageCreateRequest(BaseModel):
    chat_id: int
    is_user: bool
    content: str

class MessageResponse(BaseModel):
    message_id: int
    chat_id: int
    is_user: bool
    content: str
    created_at: datetime

    class Config:
        from_attributes = True