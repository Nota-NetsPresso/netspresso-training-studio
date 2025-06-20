from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from src.api.deps import get_token
from src.api.v1.schemas.model import ModelDetailResponse, ModelsResponse, ModelUrlResponse
from src.api.v1.schemas.tasks.benchmark.benchmark_task import BenchmarksResponse
from src.api.v1.schemas.tasks.compression.compression_task import CompressionsResponse
from src.api.v1.schemas.tasks.conversion.conversion_task import ConversionsResponse
from src.api.v1.schemas.tasks.evaluation.evaluation_task import EvaluationsResponse
from src.api.v1.schemas.user import Token
from src.core.db.session import get_db
from src.enums.task import RetrievalTaskType
from src.services.benchmark_task import benchmark_task_service
from src.services.compression_task import compression_task_service
from src.services.conversion_task import conversion_task_service
from src.services.evaluation_task import evaluation_task_service
from src.services.model import model_service

router = APIRouter()


@router.get("", response_model=ModelsResponse)
def get_models(
    *,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
    task_type: Optional[RetrievalTaskType] = Query(None, description="Filter models by task type"),
    project_id: Optional[str] = Query(None, description="Filter models by project ID"),
) -> ModelsResponse:
    models = model_service.get_models(
        db=db,
        token=token.access_token,
        task_type=task_type,
        project_id=project_id
    )

    return ModelsResponse(data=models, total_count=len(models))


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ModelDetailResponse:
    model = model_service.get_model(db=db, model_id=model_id, token=token.access_token)

    return ModelDetailResponse(data=model)


@router.delete("/{model_id}", response_model=ModelDetailResponse)
def delete_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ModelDetailResponse:
    model = model_service.delete_model(db=db, model_id=model_id, token=token.access_token)

    return ModelDetailResponse(data=model)


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ModelUrlResponse:
    presigned_url = model_service.download_model(db=db, model_id=model_id, token=token.access_token)

    return ModelUrlResponse(data=presigned_url)


@router.get("/{model_id}/compressions", response_model=CompressionsResponse)
def get_model_compression_tasks(
    model_id: str = Path(..., description="Model ID to get all related compression tasks"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> CompressionsResponse:
    compression_tasks = compression_task_service.get_compression_tasks(
        db=db,
        model_id=model_id,
        token=token.access_token,
    )

    return CompressionsResponse(data=compression_tasks, result_count=len(compression_tasks), total_count=len(compression_tasks))


@router.get("/{model_id}/conversions", response_model=ConversionsResponse)
def get_model_conversion_tasks(
    model_id: str = Path(..., description="Model ID to get all related conversion tasks"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ConversionsResponse:
    conversion_tasks = conversion_task_service.get_conversion_tasks(
        db=db,
        model_id=model_id,
        token=token.access_token,
    )

    return ConversionsResponse(data=conversion_tasks, result_count=len(conversion_tasks), total_count=len(conversion_tasks))


@router.get("/{model_id}/benchmarks", response_model=BenchmarksResponse)
def get_model_benchmark_tasks(
    model_id: str = Path(..., description="Model ID to get all related benchmark tasks"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> BenchmarksResponse:
    benchmark_tasks = benchmark_task_service.get_benchmark_tasks(
        db=db,
        model_id=model_id,
        token=token.access_token,
    )

    return BenchmarksResponse(data=benchmark_tasks, result_count=len(benchmark_tasks), total_count=len(benchmark_tasks))


@router.get("/{model_id}/evaluations", response_model=EvaluationsResponse)
def get_model_evaluations(
    model_id: str = Path(..., description="Model ID to get all related evaluation tasks"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> EvaluationsResponse:
    # 1. Get all converted model IDs derived from the trained model
    _, _, converted_model_ids = conversion_task_service._get_conversion_info(db, model_id)
    all_model_ids = [model_id] + converted_model_ids

    # 2. Get all evaluation tasks using original model ID and converted model IDs
    evaluation_tasks = evaluation_task_service.get_evaluation_tasks_by_ids(
        db=db,
        model_ids=all_model_ids
    )

    return EvaluationsResponse(data=evaluation_tasks, result_count=len(evaluation_tasks), total_count=len(evaluation_tasks))
