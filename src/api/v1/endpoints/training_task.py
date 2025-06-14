from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_token
from src.api.v1.schemas.tasks.dataset import LocalTrainingDatasetsResponse
from src.api.v1.schemas.tasks.hyperparameter import (
    SupportedModelResponse,
    SupportedOptimizersResponse,
    SupportedSchedulersResponse,
)
from src.api.v1.schemas.tasks.training_task import TrainingCreate, TrainingCreateResponse, TrainingResponse
from src.api.v1.schemas.user import Token
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
def start_training_task(
    request_body: TrainingCreate,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> TrainingCreateResponse:
    training_task = training_task_service.create_training_task(db=db, training_in=request_body, token=token.access_token)
    training_task_payload = training_task_service.start_training_task(
        db=db,
        training_in=request_body,
        training_task=training_task,
        token=token.access_token,
    )

    return TrainingCreateResponse(data=training_task_payload)


@router.get("/trainings/{task_id}", response_model=TrainingResponse)
def get_training_task(
    *,
    task_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> TrainingResponse:
    training_task = training_task_service.get_training_task(db=db, task_id=task_id, token=token.access_token)

    return TrainingResponse(data=training_task)


@router.get("/trainings/datasets/local", response_model=LocalTrainingDatasetsResponse)
def get_training_datasets() -> LocalTrainingDatasetsResponse:
    training_datasets = training_task_service.get_training_datasets_from_local()

    return LocalTrainingDatasetsResponse(data=training_datasets)
