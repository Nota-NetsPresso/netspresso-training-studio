from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from src.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from src.api.v1.schemas.project import ProjectSimplePayload
from src.enums.task import TaskStatus


class ExperimentStatus(BaseModel):
    compress: TaskStatus = Field(default=TaskStatus.NOT_STARTED, description="The status of the compression experiment.")
    convert: TaskStatus = Field(default=TaskStatus.NOT_STARTED, description="The status of the conversion experiment.")
    benchmark: TaskStatus = Field(default=TaskStatus.NOT_STARTED, description="The status of the benchmark experiment.")
    evaluate: TaskStatus = Field(default=TaskStatus.NOT_STARTED, description="The status of the evaluation experiment.")


class ModelPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., description="The unique identifier for the model.")
    name: str = Field(..., description="The name of the model.")
    type: str = Field(..., description="The type of the model (e.g., trained_model, compressed_model).")
    is_retrainable: bool
    status: TaskStatus = Field(default=TaskStatus.NOT_STARTED, description="The current status of the model.")
    train_task_id: Optional[str] = None
    project_id: str
    project: ProjectSimplePayload
    user_id: str
    compress_task_ids: Optional[List] = []
    convert_task_ids: Optional[List] = []
    benchmark_task_ids: Optional[List] = []
    evaluation_task_ids: Optional[List] = []
    is_deleted: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    latest_experiments: ExperimentStatus = Field(default_factory=ExperimentStatus)


class ExperimentStatusResponse(ResponseItem):
    data: ExperimentStatus


class ModelDetailResponse(ResponseItem):
    data: ModelPayload


class ModelsResponse(ResponsePaginationItems):
    data: List[ModelPayload]


class PresignedUrl(BaseModel):
    model_id: str = Field(..., description="model_id for upload")
    file_name: str = Field(..., description="file name")
    presigned_url: HttpUrl = Field(..., description="presigned model url")


class ModelUrlResponse(ResponseItem):
    data: Optional[PresignedUrl] = PresignedUrl
