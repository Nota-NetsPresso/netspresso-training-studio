from fastapi import status

from src.enums.exception import Origin
from src.enums.task import TaskType
from src.exceptions.base import AdditionalData, ExceptionBase


class TaskNotFoundException(ExceptionBase):
    def __init__(self, task_type: TaskType):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code=f"{task_type.name}40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=f"The {task_type} task does not exist.",
        )


class TaskIsDeletedException(ExceptionBase):
    def __init__(self, task_type: TaskType, task_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code=f"{task_type.name}40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The {task_type} task with ID '{task_id}' has been already deleted.",
        )
