from .BaseDataModel import BaseDataModel
from .db_schemes import Message
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy import select, func, desc

class MessageModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client
    
    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance
    
    async def create_message(self, message: Message):
        async with self.db_client() as session:
            async with session.begin():
                session.add(message)
            await session.commit()
            await session.refresh(message)
        
        return message.to_dict()

    async def get_messages_by_chat_id(self, chat_id: int, page: int = 1, page_size: int = 10):

        async with self.db_client() as session:
            async with session.begin():
                total_messages = await session.execute(
                    select(func.count(Message.message_id)).where(Message.chat_id == chat_id)
                )
                total_messages = total_messages.scalar_one()

                total_pages = total_messages // page_size
                if total_messages % page_size > 0:
                    total_pages += 1

                query = (
                    select(Message)
                    .where(Message.chat_id == chat_id)
                    .order_by(desc(Message.created_at))  # En yeni mesajlar önce

                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
                result = await session.execute(query)
                messages = [message.to_dict() for message in result.scalars().all()]

        return messages, total_pages

    async def delete_message(self, message_id: int):
        async with self.db_client() as session:
            async with session.begin():
                message = await session.get(Message, message_id)
                if message:
                    await session.delete(message)
                    await session.commit()
                    return True
                else:
                    return False
    async def get_message_by_id(self, message_id: int):
        async with self.db_client() as session:
            async with session.begin():
                message = await session.get(Message, message_id)
                if message:
                    return message.to_dict() 
                else:
                    return None
                    
    async def update_message(self, message_id: int, content: str):
        async with self.db_client() as session:
            async with session.begin():
                message = await session.get(Message, message_id)
                if message:
                    message.content = content
                    await session.commit()
                    return message
                else:
                    return None