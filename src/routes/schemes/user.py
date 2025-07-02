from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class UserResponse(BaseModel):
    user_id: int
    user_uuid: UUID
    firebase_uid: str
    email: str
    created_at: datetime

    class Config:
        orm_mode = True
        from_attributes=True
