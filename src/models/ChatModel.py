from .BaseDataModel import BaseDataModel
from .db_schemes import Chat,Message,Project
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func
from uuid import UUID
from sqlalchemy import delete

class ChatModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance
    
    async def create_chat(self, chat: Chat):

        async with self.db_client() as session:
            async with session.begin():
                session.add(chat)
            await session.commit()
            await session.refresh(chat)
        
        chat_pydantic = chat.to_dict()
        return chat_pydantic
    
    async def get_chats_by_user_id(self, user_id: int, page: int = 1, page_size: int = 10):
        async with self.db_client() as session:
            async with session.begin():
                total_chats = await session.execute(
                    select(func.count(Chat.chat_id)).where(Chat.user_id == user_id)
                )
                total_chats = total_chats.scalar_one()

                total_pages = total_chats // page_size
                if total_chats % page_size > 0:
                    total_pages += 1

                query = (
                    select(Chat)
                    .where(Chat.user_id == user_id)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
                result = await session.execute(query)
                chats = result.scalars().all()
                chats_pydantic = [chat.to_dict() for chat in chats]

        return chats_pydantic, total_pages

    async def delete_chat(self, chat_id: int):
        async with self.db_client() as session:
            async with session.begin():
                # Get the chat first
                chat = await session.get(Chat, chat_id)
                if not chat:
                    return False
                # Delete related messages
                await session.execute(
                    delete(Message).where(Message.chat_id == chat_id)
                )
                await session.execute(
                    delete(Project).where(Project.project_id == chat.project_id)
                )
                await session.delete(chat)
                return True

    

    async def get_chat_by_uuid(self, chat_uuid: UUID):
        async with self.db_client() as session:
            async with session.begin():
                result = await session.execute(
                    select(Chat).where(Chat.chat_uuid == chat_uuid))
                chat = result.scalar_one_or_none()
            return chat.to_dict() if chat else None
    
    async def get_chat_by_id(self, chat_id: int):
        async with self.db_client() as session:
            async with session.begin():
                chat = await session.get(Chat, chat_id)
                return chat.to_dict() if chat else None