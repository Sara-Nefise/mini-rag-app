from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ChatCreateRequest(BaseModel):
    user_id: int
    project_id: Optional[int] = None
    title: str


class ChatResponse(BaseModel):
    chat_id: int
    user_id: int
    project_id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True
    @classmethod
    def from_orm(cls, obj):
        # تحويل datetime إلى string قبل التقديم
        obj_dict = obj.__dict__
        if 'created_at' in obj_dict:
            obj_dict['created_at'] = obj_dict['created_at'].isoformat()  # تحويل datetime إلى string
        return super().from_orm(obj)


class ChatListResponse(BaseModel):
    chats: list[ChatResponse]
    total_pages: int
