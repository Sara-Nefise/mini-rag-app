from fastapi import APIRouter, Depends, HTTPException, status, Request,Header
from controllers.FirebaseController import FirebaseController
from controllers.PasswordController import PasswordController
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from models import ResponseSignal
from .schemes.auth import AuthRequest
from firebase_admin import auth
from models.UserModel import UserModel
from models.db_schemes import User

auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["api_v1", "auth"],
)

@auth_router.post("/register")
async def signup(request:Request, auth_request:AuthRequest):  
    try:
        # firebase = FirebaseController( firebase_auth_client=auth)
        # uid,token = await firebase.create_firebase_user(auth_request.email, auth_request.password)
        # if not uid or  not token:
        #     return JSONResponse(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         content={
        #             "signal": ResponseSignal.USER_CREATION_FAILED.value
        #         }
        #     )

        user_data = {
            "email": auth_request.email,
            "firebase_uid": auth_request.firebase_uid,
        }
        print("user_data", user_data)


        user_model = await UserModel.create_instance(
                        db_client=request.app.db_client)
        user = User(**user_data)

        created_user = await user_model.create_user(user=user)
        print("created_user", created_user)
        if not created_user:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.USER_CREATION_FAILED.value
                }
            )
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "signal": ResponseSignal.USER_CREATED_SUCCESS.value,
                "email": created_user.email,
                "user_id": created_user.user_id,
                # "token": token,
            }
        )
    except Exception as e:
        print("Error in user creation:", str(e))
        if 'EMAIL_EXISTS' in str(e):  # Check if the error is because the email already exists
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "signal": "user_creation_failed",
                    "detail": "The user with the provided email already exists"
                }
            )
        # Handle other Firebase errors
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "user_creation_failed",
                "detail": str(e)
            }
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.USER_CREATION_FAILED.value,
                "detail": str(e)
            }
        )

@auth_router.post("/verify")
async def verify_token(authorization: str = Header(...)):
    try:
        token ="eyJhbGciOiJSUzI1NiIsImtpZCI6IjU5MWYxNWRlZTg0OTUzNjZjOTgyZTA1MTMzYmNhOGYyNDg5ZWFjNzIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2hhdC1hcHAtNjk3M2MiLCJhdWQiOiJjaGF0LWFwcC02OTczYyIsImF1dGhfdGltZSI6MTc0Njg5NTAwNiwidXNlcl9pZCI6IkZYWUpZTlFEb0pkQ0RkSVFiQ0FJb3BMY3JOMjIiLCJzdWIiOiJGWFlKWU5RRG9KZENEZElRYkNBSW9wTGNyTjIyIiwiaWF0IjoxNzQ2ODk1MDA2LCJleHAiOjE3NDY4OTg2MDYsImVtYWlsIjoib3BvQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJvcG9AZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoicGFzc3dvcmQifX0.zJL3uBipGs_ITzK_JDa-koeBaYdHl_Fp4R-WStElX9Ol2xHGk_uU4ZJV21xFymkQIqFWC0p9z2KZHSW_MbhOwqhv8JWDhzvx92IcRiH7dMEDZHCZXVPTuo6hnDTukQEaiE7MjpZbBHF9k6Z0buisa91Ohmm-KLLS5Q5lXMgxP8L6l1deJA4mlHF0U1xv5x_vVbkd-ScVUCW9tGMHOSVMIfSoSFHnpzMr3IRPiua15y4xZS8yJBTDwkseLkfid4J_slhES24eii5w8a6sYc9H-OOjcl5SnvUnTs3jQhQdXTB9gDcXM4BjGgzDjoNji130S3tww_fqObOyaTjbqGvo5Q"
        #  authorization.split("Bearer ")[1]
        firebase = FirebaseController(firebase_auth_client=auth)
        payload = await firebase.verify_token(id_token=token)
        return JSONResponse(
            content={
                "signal": ResponseSignal.TOKEN_VERIFIED.value,
                "decoded_token": payload
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "signal": ResponseSignal.INVALID_TOKEN.value,
                "detail": str(e)
            }
        )
