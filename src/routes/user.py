from fastapi import APIRouter, Depends, HTTPException, status,Header,Request
from sqlalchemy.orm import Session
from models.db_schemes import User
from typing import List
from models.UserModel import UserModel
from fastapi.responses import JSONResponse
from models import ResponseSignal
from controllers.FirebaseController import FirebaseController
from firebase_admin import auth
import requests
from .schemes.user import UserResponse
from helpers.auth_dependencies import verify_firebase_token

user_router = APIRouter(
    prefix="/api/v1/user",
    tags=["api_v1", "user"],
    # dependencies=[Depends(verify_firebase_token)]
)

@user_router.get("/profile", summary="Get user profile from Firebase token")
async def get_user_profile(request: Request, authorization: str = Header(...)):
    try:
        id_token = authorization.replace("Bearer ", "").strip()

        firebase_controller = FirebaseController(firebase_auth_client=auth)
        payload = await firebase_controller.verify_token(id_token=id_token)
        if not payload:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"signal": ResponseSignal.TOKEN_NOT_VERIFIED.value}
            )

        firebase_uid = payload.get("uid")
        if not firebase_uid:
            raise HTTPException(status_code=400, detail="Invalid token payload")

        user_model = await UserModel.create_instance(db_client=request.app.db_client)
        user = await user_model.get_user_by_firebase_uid(firebase_uid)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"signal": ResponseSignal.USER_NOT_FOUND.value}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.USER_FETCHED_SUCCESS.value,
                "user": {
                    "email": user.email,
                    "user_id": user.user_id,
                    "firebase_uid": user.firebase_uid,
                    "created_at": str(user.created_at)
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )

@user_router.get("/{user_id}", summary="Get user by ID")
async def get_user_by_id(request: Request, user_id: int):
    try:
        user_model = await UserModel.create_instance(db_client=request.app.db_client)
        user = await user_model.get_user_by_id(user_id)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"signal": ResponseSignal.USER_NOT_FOUND.value}
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.USER_FETCHED_SUCCESS.value,
                "data":user  
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )
