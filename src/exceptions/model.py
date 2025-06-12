from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class ModelNotFoundException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="model40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message="The model does not exist.",
        )


class ModelIsDeletedException(ExceptionBase):
    def __init__(self, model_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="model40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The model with ID '{model_id}' has been already deleted.",
        )


class ModelCannotBeDeletedException(ExceptionBase):
    def __init__(self, model_id: str):
        message = f"The model with ID '{model_id}' cannot be deleted. Only trained and compressed models can be deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="model40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )
