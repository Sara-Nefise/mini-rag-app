from fastapi import APIRouter, Depends, HTTPException, status, Request,Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from models.db_schemes import Chat
from models.ChatModel import ChatModel
from models.ProjectModel import ProjectModel
from models.UserModel import UserModel
from typing import List
from models import ResponseSignal
from fastapi.responses import JSONResponse
from .schemes.chat import ChatCreateRequest, ChatResponse
from uuid import UUID
from firebase_admin import auth
from controllers.FirebaseController import FirebaseController
from helpers.auth_dependencies import verify_firebase_token

chat_router = APIRouter(
    prefix="/api/v1/chat",
    tags=["api_v1", "chat"],
    # dependencies=[Depends(verify_firebase_token)]

)
@chat_router.post("/")
async def create_chat(request:Request,chat: ChatCreateRequest):

        # Step 0: Check if user exists
        user_model = await UserModel.create_instance(db_client=request.app.db_client)
        user = await user_model.get_user_by_id(chat.user_id)
        
        if not user:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.USER_NOT_FOUND.value,
                    "message": f"User with ID {chat.user_id} not found"
                }
            )
        # Step 1: Create project
        project_model = await ProjectModel.create_instance(
            db_client=request.app.db_client
        )
        project_id = await project_model.create_empty_project()
        chat.project_id = project_id

        chat_instance = Chat(**chat.dict())

        # Step 2: Create chat with project_id
        chat_model = await ChatModel.create_instance(request.app.db_client)
        created_chat = await chat_model.create_chat(chat_instance)

        if not created_chat:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.CHAT_CREATION_FAILED.value
                }
            )        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "data":created_chat,
            "signal": ResponseSignal.CHAT_CREATED_SUCCESS.value,
            }
        )

@chat_router.get("/user/{user_id}")
async def get_chats_by_user(request:Request,user_id: int, page: int = 1, page_size: int = 10):
    chat_model = await ChatModel.create_instance(db_client=request.app.db_client)
    try:
        # Step 0: Check if user exists
        user_model = await UserModel.create_instance(db_client=request.app.db_client)
        user = await user_model.get_user_by_id(user_id)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.USER_NOT_FOUND.value,
                    "message": f"User with ID {user_id} not found"
                }
            )
        # Step 1: Get chats by user_id
        chats, total_pages = await chat_model.get_chats_by_user_id(user_id, page, page_size)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "chats": chats,
                "total_pages": total_pages,
                "signal": ResponseSignal.CHATS_FETCHED_SUCCESS.value,
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

@chat_router.get("/{chat_id}")
async def get_chat_by_id(request:Request,chat_id: int):
    chat_model = await ChatModel.create_instance(request.app.db_client)
    try:
        chat = await chat_model.get_chat_by_id(chat_id)
        if not chat:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.CHAT_NOT_FOUND.value,
                }
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "data": chat,
                "signal": ResponseSignal.CHAT_FETCHED_SUCCESS.value,
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

@chat_router.delete("/{chat_id}")
async def delete_chat(request:Request,chat_id: int):
    chat_model = await ChatModel.create_instance(db_client=request.app.db_client)
    try:
        success = await chat_model.delete_chat(chat_id)
        if not success:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.CHAT_NOT_FOUND.value,
                }
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"signal": ResponseSignal.CHAT_DELETED_SUCCESS.value}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )


