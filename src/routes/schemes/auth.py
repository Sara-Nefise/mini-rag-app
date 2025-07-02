from pydantic import BaseModel
from datetime import datetime


class AuthRequest(BaseModel):
    email: str
    firebase_uid: str
