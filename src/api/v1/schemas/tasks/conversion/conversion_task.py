from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from src.api.v1.schemas.tasks.common.device import (
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    TargetDevicePayload,
    TargetFrameworkPayload,
)
from src.enums.conversion import EvaluationTargetFramework, PrecisionForConversion, TargetFramework
from src.enums.device import DeviceName, SoftwareVersion


class ConversionForEvaluationCreate(BaseModel):
    framework: EvaluationTargetFramework = Field(description="Framework name")
    device_name: Optional[DeviceName] = Field(default=None, description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    precision: Optional[PrecisionForConversion] = Field(default=None, description="Precision")


class ConversionCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    framework: TargetFramework = Field(description="Framework name")
    device_name: DeviceName = Field(description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    precision: PrecisionForConversion = Field(description="Precision")
    calibration_dataset_path: Optional[str] = Field(default=None, description="Path to the calibration dataset")


class ConversionPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    framework: TargetFrameworkPayload
    device: TargetDevicePayload
    software_version: Optional[SoftwareVersionPayload] = None
    precision: PrecisionForConversionPayload
    status: str
    is_deleted: bool
    error_detail: Optional[Dict] = None
    input_model_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversionCreatePayload(BaseModel):
    task_id: str


class ConversionCreateResponse(ResponseItem):
    data: ConversionCreatePayload


class ConversionResponse(ResponseItem):
    data: ConversionPayload


class ConversionsResponse(ResponsePaginationItems):
    data: List[ConversionPayload]
    result_count: int
    total_count: int
