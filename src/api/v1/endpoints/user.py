from fastapi import APIRouter, Depends

from src.api.deps import get_token
from src.api.v1.schemas.user import Token, UserResponse
from src.services.user import user_service

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def get_user(
    token: Token = Depends(get_token),
) -> UserResponse:
    user = user_service.get_user_info(token=token.access_token)

    return UserResponse(data=user)
