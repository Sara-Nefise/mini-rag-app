from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, Integer, DateTime, func, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import UUID

import uuid

class User(SQLAlchemyBase):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user__uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    firebase_uid = Column(String(128), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chats = relationship("Chat", back_populates="user", )

    __table_args__ = (
        Index('ix_user_firebase_uid', firebase_uid),
        Index('ix_user_email', email),
    )
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "user_uuid": str(self.user__uuid),
            "firebase_uid": self.firebase_uid,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }