from fastapi import status

from src.enums.exception import Origin
from src.exceptions.base import AdditionalData, ExceptionBase


class NotValidChannelAxisRangeException(ExceptionBase):
    def __init__(self, reshape_channel_axis: int):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The reshape_channel_axis value must be in the range [0, 1, -1, -2], but got {reshape_channel_axis}",
        )


class EmptyCompressionParamsException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message="The available_layer values are all empty. Please provide values for compression.",
        )


class NotValidSlampRatioException(ExceptionBase):
    def __init__(self, ratio: float):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression40003",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The ratio range for SLAMP must be between 0 and 1 (exclusive), but got {ratio}",
        )


class NotValidVbmfRatioException(ExceptionBase):
    def __init__(self, ratio: float):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression40004",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"The ratio range for VBMF must be between -1 and 1 (inclusive), but got {ratio}",
        )


class NotFillInputLayersException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression40005",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message="Input Layers fields must be filled",
        )


class FailedUploadModelException(ExceptionBase):
    def __init__(self):
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="compression50001",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message="Failed to upload compressed model",
        )
