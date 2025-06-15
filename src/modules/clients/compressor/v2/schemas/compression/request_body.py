from dataclasses import dataclass, field
from typing import List

from src.clients.compressor.v2.schemas.compression.base import Layer, Options, RecommendationOptions
from src.enums.compression import CompressionMethod, RecommendationMethod
from src.exceptions.compressor import (
    EmptyCompressionParamsException,
    NotValidSlampRatioException,
    NotValidVbmfRatioException,
)


@dataclass
class RequestCreateCompression:
    ai_model_id: str
    compression_method: CompressionMethod
    options: Options = field(default_factory=Options)


@dataclass
class RequestCreateRecommendation:
    recommendation_method: RecommendationMethod
    recommendation_ratio: float
    options: RecommendationOptions = field(default_factory=RecommendationOptions)

    def __post_init__(self):
        if self.recommendation_method in [RecommendationMethod.SLAMP] and not 0 < self.recommendation_ratio < 1:
            raise NotValidSlampRatioException(ratio=self.recommendation_ratio)
        elif self.recommendation_method in [RecommendationMethod.VBMF] and not -1 <= self.recommendation_ratio <= 1:
            raise NotValidVbmfRatioException(ratio=self.recommendation_ratio)


@dataclass
class RequestUpdateCompression:
    available_layers: List[Layer]
    options: Options = field(default_factory=Options)

    def __post_init__(self):
        if all(not available_layer.values for available_layer in self.available_layers):
            raise EmptyCompressionParamsException()


@dataclass
class RequestAutomaticCompressionParams:
    compression_ratio: float = 0.5


@dataclass
class RequestAvailableLayers:
    compression_method: CompressionMethod
    options: Options = field(default_factory=Options)
