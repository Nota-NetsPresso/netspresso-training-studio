from fastapi import status

from src.enums.exception import Origin
from src.enums.model import ModelType
from src.exceptions.base import AdditionalData, ExceptionBase


class InvalidBenchmarkModelException(ExceptionBase):
    def __init__(self, model_id: str, model_type: ModelType):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="benchmark40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"Model {model_id} is not a trained or compressed model. Model type: {model_type}"
        )
