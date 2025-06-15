from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class EvaluationTaskNotFoundException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message="The evaluation task does not exist.",
        )


class EvaluationTaskIsDeletedException(ExceptionBase):
    def __init__(self, task_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The evaluation task with ID '{task_id}' has been already deleted.",
        )


class EvaluationTaskAlreadyExistsException(ExceptionBase):
    def __init__(self, task_id: str, task_status: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation40901",
            status_code=status.HTTP_409_CONFLICT,
            name=self.__class__.__name__,
            message=f"Evaluation task already exists with ID '{task_id}' (status: {task_status}).",
        )


class EvaluationTaskInProgressException(ExceptionBase):
    def __init__(self, task_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The evaluation task with ID '{task_id}' is currently in progress.",
        )


class EvaluationTaskFailedException(ExceptionBase):
    def __init__(self, task_id: str, error_message: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=f"The evaluation task with ID '{task_id}' failed: {error_message}",
        )


class InvalidEvaluationDatasetException(ExceptionBase):
    def __init__(self, dataset_id: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="evaluation40003",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The dataset with ID '{dataset_id}' is not valid for evaluation.",
        )


class UnsupportedEvaluationFrameworkException(PyNPException):
    def __init__(self, framework: TargetFramework):
        message = f"The framework '{framework}' is not supported for evaluation."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="EVALUATION40002",
            name=self.__class__.__name__,
            message=message,
        )


class EvaluationResultFileNotFoundException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The evaluation result file for task '{task_id}' does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, task_id=task_id),
            error_code="EVALUATION40404",
            name=self.__class__.__name__,
            message=message,
        )


class EvaluationDownloadURLGenerationException(PyNPException):
    def __init__(self, task_id: str, error_details: str):
        message = f"Failed to generate download URL for evaluation task '{task_id}': {error_details}"
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, task_id=task_id, error_details=error_details),
            error_code="EVALUATION50000",
            name=self.__class__.__name__,
            message=message,
        )
