from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import api_key_header
from src.api.v1.schemas.tasks.dataset import LocalTrainingDatasetsResponse
from src.api.v1.schemas.tasks.hyperparameter import (
    SupportedModelResponse,
    SupportedOptimizersResponse,
    SupportedSchedulersResponse,
)
from src.api.v1.schemas.tasks.training_task import TrainingCreate, TrainingCreateResponse, TrainingResponse
from src.core.db.session import get_db
from src.services.training_task import training_task_service

router = APIRouter()


@router.get(
    "/trainings/configuration/models",
    response_model=SupportedModelResponse,
    description="Get supported models for training tasks.",
)
def get_supported_models() -> SupportedModelResponse:
    supported_models = training_task_service.get_supported_models()

    return SupportedModelResponse(data=supported_models)


@router.get(
    "/trainings/configuration/optimizers",
    response_model=SupportedOptimizersResponse,
    description="Get supported optimizers for training tasks.",
)
def get_supported_optimizers() -> SupportedOptimizersResponse:
    supported_optimizers = training_task_service.get_supported_optimizers()

    return SupportedOptimizersResponse(data=supported_optimizers)


@router.get(
    "/trainings/configuration/schedulers",
    response_model=SupportedSchedulersResponse,
    description="Get supported schedulers for training tasks.",
)
def get_supported_schedulers() -> SupportedSchedulersResponse:
    supported_schedulers = training_task_service.get_supported_schedulers()

    return SupportedSchedulersResponse(data=supported_schedulers)


@router.post("/trainings", response_model=TrainingCreateResponse, status_code=201)
def create_training_task(
    request_body: TrainingCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainingCreateResponse:
    training_task = training_task_service.create_training_task(db=db, training_in=request_body, api_key=api_key)

    return TrainingCreateResponse(data=training_task)


@router.get("/trainings/{task_id}", response_model=TrainingResponse)
def get_training_task(
    *,
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainingResponse:
    training_task = training_task_service.get_training_task(db=db, task_id=task_id, api_key=api_key)

    return TrainingResponse(data=training_task)


@router.get("/trainings/datasets/local", response_model=LocalTrainingDatasetsResponse)
def get_training_datasets() -> LocalTrainingDatasetsResponse:
    training_datasets = training_task_service.get_training_datasets_from_local()

    return LocalTrainingDatasetsResponse(data=training_datasets)
