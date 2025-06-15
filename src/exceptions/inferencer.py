from typing import List

from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class NotSupportedSuffixException(ExceptionBase):
    def __init__(self, available_suffixes: List, suffix: str):
        message = f"The suffix supports {available_suffixes}. The entered suffix is {suffix}."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )
