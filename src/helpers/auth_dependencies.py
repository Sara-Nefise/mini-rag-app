
from fastapi import Request, HTTPException, status, Depends
from firebase_admin import auth
from controllers.FirebaseController import FirebaseController
from helpers.config import get_settings, Settings


async def verify_firebase_token(request: Request, app_settings: Settings = Depends(get_settings)):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

        token = auth_header.split(" ")[1]
        try:
            firebase = FirebaseController( firebase_auth_client=auth)
            print(token)
            
            # Remove this
            decoded_token=await firebase.get_firebase_id_token(id=token,api_key=app_settings.FIREBASE_API_KEY)
           
            # Uncomment this
            # decoded_token = auth.verify_id_token(token)
            request.state.user = decoded_token  
            return decoded_token
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
