from fastapi import APIRouter, Depends, Path, Query
from loguru import logger
from sqlalchemy.orm import Session

from src.api.deps import api_key_header, get_token
from src.api.v1.schemas.tasks.common.dataset import EvaluationDatasetPayload
from src.api.v1.schemas.tasks.common.device import SupportedDevicesResponse
from src.api.v1.schemas.tasks.evaluation.evaluation_task import (
    EvaluationCreate,
    EvaluationCreatePayload,
    EvaluationCreateResponse,
    EvaluationDatasetsPayload,
    EvaluationDatasetsResponse,
    EvaluationResponse,
    EvaluationResultsResponse,
    EvaluationsResponse,
)
from src.api.v1.schemas.user import Token
from src.core.db.session import get_db
from src.enums.conversion import SourceFramework
from src.services.conversion_task import conversion_task_service
from src.services.evaluation_task import evaluation_task_service

router = APIRouter()


@router.get(
    "/evaluations/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices and frameworks for model evaluation based on the source framework.",
)
def get_supported_evaluation_devices(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    framework = SourceFramework.ONNX
    supported_devices = evaluation_task_service.get_supported_devices(
        db=db, framework=framework, api_key=api_key
    )

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/evaluations", response_model=EvaluationCreateResponse, status_code=201)
def create_evaluations_task(
    request_body: EvaluationCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationCreateResponse:
    try:
        existing_task = evaluation_task_service.check_evaluation_task_status(
            db=db,
            model_id=request_body.input_model_id,
            dataset_path=request_body.dataset_path,
            confidence_score=request_body.confidence_scores,
        )
        if existing_task:
            payload = EvaluationCreatePayload(task_id=existing_task.task_id)
            return EvaluationCreateResponse(data=payload)

        evaluation_task_id = evaluation_task_service.create_evaluation_task(
            db=db, evaluation_in=request_body, api_key=api_key
        )

        response_data = EvaluationCreatePayload(task_id=evaluation_task_id)
        return EvaluationCreateResponse(data=response_data)

    except Exception as e:
        logger.error(f"Error starting evaluation task: {e}")
        raise e


# TODO: Will be removed after the evaluation task is migrated to the new task system
@router.get("/evaluations/{model_id}", response_model=EvaluationsResponse, status_code=200)
def get_evaluation_tasks(
    model_id: str = Path(..., description="Trained Model ID to get all related evaluation tasks"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> EvaluationsResponse:
    # 1. Get all converted model IDs derived from the trained model
    _, _, converted_model_ids = conversion_task_service._get_conversion_info(db, model_id)

    # 2. Get all evaluation tasks using original model ID and converted model IDs
    evaluation_tasks = []

    # Add evaluation tasks directly linked to the original model ID
    original_model_tasks = evaluation_task_service.get_evaluation_tasks(
        db=db,
        token=token.access_token,
        model_id=model_id
    )
    evaluation_tasks.extend(original_model_tasks)

    # Add evaluation tasks for each converted model
    for converted_id in converted_model_ids:
        converted_model_tasks = evaluation_task_service.get_evaluation_tasks(
            db=db,
            token=token.access_token,
            model_id=converted_id
        )
        evaluation_tasks.extend(converted_model_tasks)

    # Remove duplicates (based on task_id)
    unique_tasks = {}
    for task in evaluation_tasks:
        unique_tasks[task.task_id] = task

    evaluation_tasks = list(unique_tasks.values())

    # Calculate total count (adjusted according to the modified logic)
    total_count = len(evaluation_tasks)

    return EvaluationsResponse(
        data=evaluation_tasks,
        result_count=len(evaluation_tasks),
        total_count=total_count
    )


@router.get("/evaluations/{model_id}/datasets", response_model=EvaluationDatasetsResponse, status_code=200)
def get_unique_evaluation_datasets(
    model_id: str = Path(..., description="Converted Model ID to get unique datasets for"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> EvaluationDatasetsResponse:
    evaluation_datasets = evaluation_task_service.get_unique_datasets_by_model_id(
        db=db,
        token=token.access_token,
        model_id=model_id
    )

    datasets = [
        EvaluationDatasetPayload.model_validate(evaluation_dataset)
        for evaluation_dataset in evaluation_datasets
    ]

    response_data = EvaluationDatasetsPayload(
        model_id=model_id,
        datasets=datasets
    )

    return EvaluationDatasetsResponse(data=response_data)


@router.get("/evaluations/{model_id}/datasets/{dataset_id}/results", response_model=EvaluationResultsResponse, status_code=200)
def get_evaluation_results(
    model_id: str = Path(..., description="Converted Model ID"),
    dataset_id: str = Path(..., description="Dataset ID"),
    start: int = Query(0, description="Pagination start index"),
    size: int = Query(20, description="Page size (number of images)"),
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> EvaluationResultsResponse:
    """Get detailed evaluation results for a specific model and dataset, including predictions and result images."""
    evaluation_result = evaluation_task_service.get_evaluation_result_details(
        db=db,
        token=token.access_token,
        model_id=model_id,
        dataset_id=dataset_id,
        start=start,
        size=size
    )

    return EvaluationResultsResponse(data=evaluation_result)


@router.delete("/evaluations/{task_id}", response_model=EvaluationResponse)
def delete_evaluation_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationResponse:
    evaluation_task = evaluation_task_service.delete_evaluation_task(
        db=db,
        task_id=task_id,
        api_key=api_key
    )

    return EvaluationResponse(data=evaluation_task)


@router.delete("/evaluations/datasets/{dataset_id}", response_model=EvaluationResponse)
def delete_evaluation_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationResponse:
    evaluation_task = evaluation_task_service.delete_evaluation_dataset(
        db=db,
        dataset_id=dataset_id,
        api_key=api_key
    )

    return EvaluationResponse(data=evaluation_task)
