from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class ProjectNotFoundException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="project40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message="The project does not exist.",
        )


class ProjectIsDeletedException(ExceptionBase):
    def __init__(self, project_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="project40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The project with ID '{project_id}' has been already deleted.",
        )


class ProjectNameTooLongException(ExceptionBase):
    def __init__(self, max_length: int, actual_length: int):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="project40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The project_name exceeds maximum length. Max: {max_length}, Actual: {actual_length}",
        )


class ProjectAlreadyExistsException(ExceptionBase):
    def __init__(self, project_name: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="project40901",
            status_code=status.HTTP_409_CONFLICT,
            name=self.__class__.__name__,
            message=f"The project_name '{project_name}' already exists.",
        )
