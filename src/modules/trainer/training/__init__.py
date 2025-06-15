from src.modules.trainer.training.environment import EnvironmentConfig
from src.modules.trainer.training.logging import LoggingConfig
from src.modules.trainer.training.training import (
    ClassificationScheduleConfig,
    DetectionScheduleConfig,
    ScheduleConfig,
    SegmentationScheduleConfig,
)

TRAINING_CONFIG_TYPE = {
    "classification": ClassificationScheduleConfig,
    "detection": DetectionScheduleConfig,
    "segmentation": SegmentationScheduleConfig,
}


__all__ = ["ScheduleConfig", "TRAINING_CONFIG_TYPE", "EnvironmentConfig", "LoggingConfig"]
