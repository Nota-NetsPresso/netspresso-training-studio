from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class LinkType(str, Enum):
    DOCS = "docs"
    CONTACT = "contact"


@dataclass
class LinkInfo:
    type: LinkType = field(metadata={"description": "Link type"})
    value: str = field(metadata={"description": "Link value"})


class NotEnoughCreditException(ExceptionBase):
    def __init__(self, current_credit: int, service_credit: int, service_task_name: str):
        error_log = (
            f"Your current balance of {current_credit} credits is insufficient to complete the task.\n"
            f"{service_credit} credits are required for one {service_task_name} task.\n"
            f"For additional credit, please contact us at netspresso@nota.ai."
        )
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE, error_log=error_log),
            error_code="credit40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message="Not enough credits. Please check your credit balance.",
        )


class NotSupportedFrameworkException(ExceptionBase):
    def __init__(self, available_frameworks: List, framework: int):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE),
            error_code="framework40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"Framework {framework} is not supported. Available frameworks are: {available_frameworks}",
        )


class NotValidInputModelPathException(ExceptionBase):
    def __init__(self, path: str):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE),
            error_code="model40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The input model path '{path}' should be a file, not a directory. Example: './model/sample_model.pt'",
        )


class GatewayTimeoutException(ExceptionBase):
    def __init__(self, error_log: str):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE, error_log=error_log),
            error_code="gateway50401",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            name=self.__class__.__name__,
            message="504 Gateway Timeout: The server did not receive a timely response.",
        )


class UnexpectedException(ExceptionBase):
    def __init__(self, error_log: str, status_code: int):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE, error_log=error_log),
            error_code="server50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=f"An unexpected error occurred with status code {status_code}",
        )


class InternalServerErrorException(ExceptionBase):
    def __init__(self, error_log: str, status_code: int):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE, error_log=error_log),
            error_code="server50002",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=f"Internal server error occurred with status code {status_code}",
        )


class UnexpetedException(ExceptionBase):
    def __init__(self, error_log: str, status_code: int):
        message = f"Unexpected error occurred with status code {status_code}"
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE, error_log=error_log),
            error_code="server50003",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=message,
        )
