

from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, Integer, DateTime, func, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import UUID

import uuid


class Message(SQLAlchemyBase):
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    message_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    chat_id = Column(Integer, ForeignKey("chats.chat_id", ondelete="CASCADE"), nullable=False)
    is_user = Column(Boolean, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chat = relationship("Chat", back_populates="messages", )

    __table_args__ = (
        Index('ix_message_chat_id', chat_id),
    )
    
    def to_dict(self):
        return {
            "message_id": self.message_id,
            "message_uuid": str(self.message_uuid),
            "chat_id": self.chat_id,
            "is_user": self.is_user,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }