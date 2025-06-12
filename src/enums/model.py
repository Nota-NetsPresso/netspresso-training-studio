from enum import Enum


class ModelType(str, Enum):
    TRAINED_MODEL = "trained_models"
    COMPRESSED_MODEL = "compressed_models"
    CONVERTED_MODEL = "converted_models"
    BENCHMARKED_MODEL = "benchmarked_models"
