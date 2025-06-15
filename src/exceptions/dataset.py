from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class DatasetException(ExceptionBase):
    """Base exception for all dataset-related exceptions."""
    def __init__(self, message: str, error_code: str = "DATASET00000", status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(
            data=AdditionalData(origin=Origin.CLIENT),
            error_code=error_code,
            status_code=status_code,
            name=self.__class__.__name__,
            message=message,
        )


class DatasetDownloadError(DatasetException):
    """Exception raised when dataset download fails."""
    def __init__(self, message: str = "Failed to download dataset"):
        super().__init__(
            message=message,
            error_code="DATASET50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class DatasetPrepareError(DatasetException):
    """Exception raised when dataset preparation fails."""
    def __init__(self, message: str = "Failed to prepare dataset"):
        super().__init__(
            message=message,
            error_code="DATASET50002",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class DatasetNotFoundError(DatasetException):
    """Exception raised when dataset is not found."""
    def __init__(self, message):
        super().__init__(
            message=message,
            error_code="DATASET40401",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class DatasetInvalidSplitError(DatasetException):
    """Exception raised when an invalid dataset split is specified."""
    def __init__(self, split: str, valid_splits: list):
        super().__init__(
            message=f"Invalid dataset split '{split}'. Valid splits are: {', '.join(valid_splits)}",
            error_code="DATASET40001",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class DatasetCorruptedError(DatasetException):
    """Exception raised when dataset files are corrupted or incomplete."""
    def __init__(self, dataset_path: str = None):
        message = "Dataset files are corrupted or incomplete"
        if dataset_path:
            message = f"Dataset files at '{dataset_path}' are corrupted or incomplete"
        super().__init__(
            message=message,
            error_code="DATASET50003",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
