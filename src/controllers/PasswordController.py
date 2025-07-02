import bcrypt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from models.db_schemes import User
from .BaseController import BaseController

class PasswordController(BaseController):
    def __init__(self):
        super().__init__()

    # Hash password securely
    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    # Verify password by comparing the stored hash with the input password
    def verify_password(self, stored_hash: str, input_password: str) -> bool:
        return bcrypt.checkpw(input_password.encode('utf-8'), stored_hash.encode('utf-8'))
