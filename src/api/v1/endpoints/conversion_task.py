from fastapi import APIRouter, Depends, Query
from loguru import logger
from sqlalchemy.orm import Session

from src.api.deps import api_key_header
from src.api.v1.schemas.tasks.common.device import SupportedDevicesResponse
from src.api.v1.schemas.tasks.conversion.conversion_task import (
    ConversionCreate,
    ConversionCreatePayload,
    ConversionCreateResponse,
    ConversionResponse,
)
from src.core.db.session import get_db
from src.enums.conversion import SourceFramework
from src.services.conversion_task import conversion_task_service

router = APIRouter()


@router.get(
    "/conversions/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices and frameworks for model conversion based on the source framework.",
)
def get_supported_conversion_devices(
    framework: SourceFramework = Query(..., description="Source framework of the model to be converted."),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    supported_devices = conversion_task_service.get_supported_devices(db=db, framework=framework, api_key=api_key)

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/conversions", response_model=ConversionCreateResponse, status_code=201)
def create_conversions_task(
    request_body: ConversionCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionCreateResponse:
    try:
        existing_task = conversion_task_service.check_conversion_task_exists(
            db=db,
            input_model_id=request_body.input_model_id,
            framework=request_body.framework,
            device_name=request_body.device_name,
            precision=request_body.precision,
            software_version=request_body.software_version,
        )
        if existing_task:
            payload = ConversionCreatePayload(task_id=existing_task.task_id)
            return ConversionCreateResponse(data=payload)

        conversion_task = conversion_task_service.create_conversion_task(db=db, conversion_in=request_body, api_key=api_key)
        conversion_task_payload = conversion_task_service.start_conversion_task(
            conversion_in=request_body,
            conversion_task=conversion_task,
            api_key=api_key,
        )

        return ConversionCreateResponse(data=conversion_task_payload)

    except Exception as e:
        logger.error(f"Error starting conversion task: {e}")
        raise e


@router.get("/conversions/{task_id}", response_model=ConversionResponse)
def get_conversions_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionResponse:
    conversion_task = conversion_task_service.get_conversion_task(db=db, task_id=task_id, api_key=api_key)

    return ConversionResponse(data=conversion_task)


@router.post("/conversions/{task_id}/cancel", response_model=ConversionResponse)
def cancel_conversion_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionResponse:
    conversion_task = conversion_task_service.cancel_conversion_task(db=db, task_id=task_id, api_key=api_key)

    return ConversionResponse(data=conversion_task)


@router.delete("/conversions/{task_id}", response_model=ConversionResponse)
def delete_conversion_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionResponse:
    conversion_task = conversion_task_service.delete_conversion_task(db=db, task_id=task_id, api_key=api_key)

    return ConversionResponse(data=conversion_task)
