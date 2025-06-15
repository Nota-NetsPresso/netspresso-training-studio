from typing import List

from fastapi import status

from src.exceptions.base import AdditionalData, ExceptionBase


class NotSupportedTaskException(ExceptionBase):
    def __init__(self, available_tasks: List, task: int):
        message = f"The task supports {available_tasks}. The entered task is {task}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class NotSetDatasetException(ExceptionBase):
    def __init__(self):
        message = "The dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class NotSetModelException(ExceptionBase):
    def __init__(self):
        message = "The model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40003",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class TaskOrYamlPathException(ExceptionBase):
    def __init__(self):
        message = "Either 'task' or 'yaml_path' must be provided, but not both."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40004",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class NotSupportedModelException(ExceptionBase):
    def __init__(self, available_models: List, model_name: str, task: str):
        message = f"The '{model_name}' model is not supported for the '{task}' task. The available models are {available_models}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40005",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class RetrainingFunctionException(ExceptionBase):
    def __init__(self):
        message = "This function is intended for retraining. Please use 'set_model_config' for model setup."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40006",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class FileNotFoundErrorException(ExceptionBase):
    def __init__(self, relative_path: str):
        message = f"The required file '{relative_path}' does not exist. Please check and make sure it is in the correct location."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )


class DirectoryNotFoundException(ExceptionBase):
    def __init__(self, relative_path: str):
        message = f"The required directory '{relative_path}' does not exist. Please check and make sure it is in the correct location."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40402",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )


class BaseDirectoryNotFoundException(ExceptionBase):
    def __init__(self, base_path: str):
        message = f"The directory '{base_path}' does not exist."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40403",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )


class FailedTrainingException(ExceptionBase):
    def __init__(self, error_log: str):
        message = "An error occurred during the training process."
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="trainer50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=message,
        )


class NotCompletedTrainingException(ExceptionBase):
    def __init__(self, training_task_id: str):
        message = f"The training task {training_task_id} is not completed."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="trainer40007",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )
