from .BaseController import BaseController
from sqlalchemy.orm import Session
from typing import Dict
import firebase_admin
from firebase_admin import credentials, auth

class UserController(BaseController):
    def __init__(self, db_session: Session, firebase_auth_client):
        super().__init__()
        self.db_session = db_session
        self.firebase_auth_client = firebase_auth_client

#     async def create_user(self, email: str, password:str, firebase_uid ) -> User:
#         existing_user = self.db_session.query(User).filter(User.firebase_uid == firebase_uid).first()
#         if existing_user:
#             return existing_user
#         new_user = User(firebase_uid=firebase_uid, email=email)
#         self.db_session.add(new_user)
#         self.db_session.commit()
#         self.db_session.refresh(new_user)
#         return new_user

#     async def verify_user(self, token: str) -> Dict:
#         try:
#             decoded_token = self.firebase_auth_client.verify_id_token(token)
#             return decoded_token
#         except Exception as e:
#             raise ValueError(f"Invalid token: {str(e)}")