from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class UnauthorizedUserAccessException(ExceptionBase):
    def __init__(self, user_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE),
            error_code="auth40301",
            status_code=status.HTTP_403_FORBIDDEN,
            name=self.__class__.__name__,
            message=f"The user with ID {user_id} is not authorized to view this information.",
        )

class InvalidApiKeyException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.ROUTER),
            error_code="auth40101",
            status_code=status.HTTP_401_UNAUTHORIZED,
            name=self.__class__.__name__,
            message="The provided API key is invalid.",
        )
