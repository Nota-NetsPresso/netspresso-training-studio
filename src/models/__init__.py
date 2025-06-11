from src.core.db.session import Base, engine
from src.models.benchmark import BenchmarkResult, BenchmarkTask
from src.models.compression import CompressionModelResult, CompressionTask
from src.models.conversion import ConversionTask
from src.models.evaluation import EvaluationDataset, EvaluationTask
from src.models.model import Model
from src.models.project import Project
from src.models.training import Augmentation, Dataset, Environment, Hyperparameter, Performance, TrainingTask

Base.metadata.create_all(engine)


__all__ = [
    "BenchmarkResult",
    "BenchmarkTask",
    "CompressionModelResult",
    "CompressionTask",
    "ConversionTask",
    "EvaluationDataset",
    "EvaluationTask",
    "Model",
    "Project",
    "Augmentation",
    "Dataset",
    "Environment",
    "Hyperparameter",
    "Performance",
    "TrainingTask",
]
