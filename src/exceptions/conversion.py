from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class ConversionTaskNotFoundException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="conversion40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message="The conversion task does not exist.",
        )


class ConversionTaskIsDeletedException(ExceptionBase):
    def __init__(self, task_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="conversion40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The conversion task with ID '{task_id}' has been already deleted.",
        )


class ConversionTaskInProgressException(ExceptionBase):
    def __init__(self, task_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="conversion40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The conversion task with ID '{task_id}' is currently in progress.",
        )


class ConversionTaskFailedException(ExceptionBase):
    def __init__(self, task_id: str, error_message: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="conversion50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=f"The conversion task with ID '{task_id}' failed: {error_message}",
        )
