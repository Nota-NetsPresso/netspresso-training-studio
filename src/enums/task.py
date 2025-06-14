from enum import Enum


class TaskType(str, Enum):
    TRAINING = "training"
    RETRAINING = "retraining"
    EVALUATION = "evaluation"
    COMPRESSION = "compression"
    CONVERSION = "conversion"
    BENCHMARK = "benchmark"


class RetrievalTaskType(str, Enum):
    COMPRESS = "compress"
    RETRAIN = "retrain"
    CONVERT = "convert"
    BENCHMARK = "benchmark"
    EVALUATE = "evaluate"


class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
