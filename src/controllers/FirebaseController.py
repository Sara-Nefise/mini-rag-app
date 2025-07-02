from .BaseController import BaseController
from sqlalchemy.orm import Session
from typing import Dict
import requests
import json
class FirebaseController(BaseController):
    def __init__(self, firebase_auth_client):
        super().__init__()
        self.firebase_auth_client = firebase_auth_client

    async def verify_token(self, id_token: str) -> Dict:
        try:
            decoded_token = self.firebase_auth_client.verify_id_token(id_token)
            return decoded_token
        except Exception as e:
            raise ValueError(f"Invalid token: {str(e)}")

    async def create_firebase_user(self, email: str, password: str) -> Dict:
        try:
            user_data = {
                "email": email,
                "password": password
            }
          
            firebase_user = self.firebase_auth_client.create_user(**user_data)
            if not firebase_user:
                return False        
           
            display_name = firebase_user.display_name
            
            token = self.firebase_auth_client.create_custom_token(firebase_user.uid).decode("utf-8")
    
            return firebase_user.uid,token 
        except Exception as e:
            raise ValueError(f"Failed to create Firebase user: {str(e)}")

            try:
                self.firebase_auth_client.update_user(firebase_uid, password=new_password)
                return True
            except Exception as e:
    
                raise ValueError(f"Failed to update password: {str(e)}")


    async def get_firebase_id_token(self,id: str, api_key: str) -> str:
        # 1. Generate custom token
        custom_token = self.firebase_auth_client.create_custom_token(id).decode("utf-8")
        
        # 2. Exchange custom token for ID token
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "token": custom_token,
            "returnSecureToken": True
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            id_token = response.json().get("idToken")
            return id_token
        else:
            raise Exception(f"Failed to exchange token: {response.text}")

        
                    
