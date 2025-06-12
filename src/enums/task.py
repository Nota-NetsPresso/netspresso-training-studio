from enum import Enum


class TaskType(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    COMPRESSION = "compression"
    CONVERSION = "conversion"
    BENCHMARK = "benchmark"


class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
