from enum import Enum


class ModelType(str, Enum):
    TRAINED_MODEL = "trained_model"
    COMPRESSED_MODEL = "compressed_model"
    CONVERTED_MODEL = "converted_model"
    BENCHMARKED_MODEL = "benchmarked_model"
