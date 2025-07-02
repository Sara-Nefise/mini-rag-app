
from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, Integer, DateTime, func, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

import  uuid


class Chat(SQLAlchemyBase):
    __tablename__ = "chats"

    chat_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    chat_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="chats")
    project = relationship("Project", back_populates="chats")
    messages = relationship("Message", back_populates="chat")

    __table_args__ = (
        Index('ix_chat_user_id', user_id),
        Index('ix_chat_project_id', project_id),
    )
    def to_dict(self):
        return {
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "chat_uuid": str(self.chat_uuid),
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
        }