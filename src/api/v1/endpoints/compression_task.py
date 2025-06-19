from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from src.api.deps import api_key_header
from src.api.v1.schemas.tasks.compression.compression_task import (
    CompressionCreate,
    CompressionCreateResponse,
    CompressionResponse,
)
from src.core.db.session import get_db
from src.services.compression_task import compression_task_service

router = APIRouter()


@router.post("/compressions", response_model=CompressionCreateResponse, status_code=201)
def create_compressions_task(
    request_body: CompressionCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionCreateResponse:
    try:
        existing_task = compression_task_service.check_compression_task_exists(db=db, compression_in=request_body)
        if existing_task:
            return CompressionCreateResponse(data=existing_task)

        compression_task = compression_task_service.create_compression_task(
            db=db, compression_in=request_body, api_key=api_key
        )

        compression_task_payload = compression_task_service.start_compression_task(
            compression_in=request_body, compression_task=compression_task, api_key=api_key
        )

        return CompressionCreateResponse(data=compression_task_payload)

    except Exception as e:
        logger.error(f"Error starting compression task: {e}")
        raise e


@router.get("/compressions/{task_id}", response_model=CompressionResponse)
def get_compression_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionResponse:
    compression_task = compression_task_service.get_compression_task(
        db=db, compression_task_id=task_id, api_key=api_key
    )

    return CompressionResponse(data=compression_task)


@router.delete("/compressions/{task_id}", response_model=CompressionResponse)
def delete_compression_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionResponse:
    compression_task = compression_task_service.delete_compression_task(db=db, task_id=task_id, api_key=api_key)

    return CompressionResponse(data=compression_task)
