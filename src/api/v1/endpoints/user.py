from fastapi import APIRouter, Depends

from src.api.deps import api_key_header

router = APIRouter()


@router.get("/me")
def get_user(*, api_key: str = Depends(api_key_header)):
    pass
