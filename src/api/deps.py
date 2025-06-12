from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from src.api.v1.schemas.user import Token
from src.clients.auth import auth_client

# Define the header key
API_KEY_NAME = "X-API-Key"

# Create a security dependency for API key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def get_token(api_key: str = Depends(api_key_header)) -> Token:
    """
    Dependency that authenticates user with API key and returns token.

    Args:
        api_key: API key from request header

    Returns:
        Token: Authentication token

    Raises:
        HTTPException: If authentication fails
    """
    try:
        return auth_client.login(api_key=api_key)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
        )
