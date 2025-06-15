from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class DatabaseOperationError(ExceptionBase):
    """Base exception for database operations"""

    def __init__(self, error_log: str = "", error_code: str = "database50001"):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, error_log=error_log),
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message="Database operation failed",
        )


class DatabaseConnectionError(ExceptionBase):
    """Exception raised when database connection fails"""

    def __init__(self, error_log: str = "", error_code: str = "database50002"):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, error_log=error_log),
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message="Failed to connect to database",
        )


class DatabaseQueryError(ExceptionBase):
    """Exception raised when a database query fails"""

    def __init__(self, error_log: str = ""):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, error_log=error_log),
            error_code="database50003",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message="Database query execution failed",
        )


class DatabaseIntegrityError(ExceptionBase):
    """Exception raised when database integrity constraints are violated"""

    def __init__(self, error_log: str = ""):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, error_log=error_log),
            error_code="database40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message="Database integrity constraint violation",
        )


class DatabaseRecordNotFoundError(ExceptionBase):
    """Exception raised when a database record is not found"""

    def __init__(self, entity: str, identifier: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="database40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=f"Database record not found: {entity} with identifier '{identifier}'",
        )


class DatabaseDuplicateEntryError(ExceptionBase):
    """Exception raised when attempting to create a duplicate entry"""

    def __init__(self, entity: str, field: str, value: str):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="database40901",
            status_code=status.HTTP_409_CONFLICT,
            name=self.__class__.__name__,
            message=f"Duplicate entry: {entity} with {field} = '{value}' already exists",
        )
