from .BaseDataModel import BaseDataModel
from .db_schemes import User
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func
from routes.schemes.user import UserResponse

class UserModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client
        
    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance
    
    async def create_user(self, user: User):
        async with self.db_client() as session:
            async with session.begin():
                session.add(user)
            await session.commit()
            await session.refresh(user)
            
        return user
    async def get_user_by_id(self, user_id: int):
        async with self.db_client() as session:
            async with session.begin():
                user = await session.get(User, user_id)
                if user:
                    pydantic_user = user.to_dict()
                    return pydantic_user 
                else:
                    return None
    
    async def get_user_by_firebase_uid(self, firebase_uid: str):
        async with self.db_client() as session:
            async with session.begin():
                user = await session.execute(
                    select(User).where(User.firebase_uid == firebase_uid)
                )
                user = user.scalars().first()
                if user:
                    return user
                else:
                    return None
    
