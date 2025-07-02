from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from models.db_schemes import Message
from models.MessageModel import MessageModel
from models.ChatModel import ChatModel

from typing import List
from fastapi.responses import JSONResponse
from .schemes.message import MessageCreateRequest, MessageResponse
from models import ResponseSignal
from helpers.auth_dependencies import verify_firebase_token

message_router = APIRouter(
    prefix="/api/v1/message",
    tags=["api_v1", "message"],
    # dependencies=[Depends(verify_firebase_token)]

)


@message_router.post("/")
async def create_message(request: Request, message:MessageCreateRequest):
    pass
    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    try:
        # Step 0: Check if the chat exists
        chat_model = await ChatModel.create_instance(db_client=request.app.db_client)
        chat = await chat_model.get_chat_by_id(message.chat_id)
        if not chat:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.CHAT_NOT_FOUND.value,
                    "message": f"Chat with ID {message.chat_id} not found"
                }
            )
        message_instance = Message(**message.dict())

        created_message = await message_model.create_message(message_instance)
        
        return JSONResponse(
                status_code=status.HTTP_200_OK,
                content= {
                    "data": created_message,
                    "signal": ResponseSignal.MESSAGE_CREATED_SUCCESS.value}  # Replace ResponseSignal with actual string value
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )

@message_router.get("/chat/{chat_id}")
async def get_messages_by_chat(request:Request,chat_id: int, page: int = 1, page_size: int = 10):
    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    try:
        messages, total_pages = await message_model.get_messages_by_chat_id(chat_id, page, page_size)
        return JSONResponse(
                status_code=status.HTTP_200_OK,
                content= {
                    "data": messages,
                    "total_pages": total_pages,
                    "signal":ResponseSignal.MESSAGES_FETCHED_SUCCESS.value}  # Replace ResponseSignal with actual string value
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )

@message_router.get("/{message_id}")
async def get_message_by_id(request:Request,message_id: int):
    message_model = await MessageModel.create_instance(db_client= request.app.db_client)
    try:
        message = await message_model.get_message_by_id(message_id)
        if not message:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.MESSAGE_NOT_FOUND.value,
                }
            )
        return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "signal":ResponseSignal.MESSAGES_FETCHED_SUCCESS.value,
                    "message": message,
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

@message_router.delete("/{message_id}")
async def delete_message(request:Request,message_id: int):
    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    try:
        success = await message_model.delete_message(message_id)
        if not success:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.MESSAGE_NOT_FOUND.value,
                }
            )
        return JSONResponse(
                status_code=status.HTTP_200_OK,
                content= {"signal": ResponseSignal.MESSAGE_DELETED_SUCCESS.value,}  # Replace ResponseSignal with actual string value
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.BAD_REQUEST.value,
                "detail": str(e)
            }
        )
