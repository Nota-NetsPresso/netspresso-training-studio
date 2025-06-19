from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from src.api.v1.schemas.tasks.common.device import (
    BenchmarkResultPayload,
    HardwareTypePayload,
    PrecisionForBenchmarkPayload,
    SoftwareVersionPayload,
    TargetDevicePayload,
    TargetFrameworkPayload,
)
from src.enums.device import DeviceName, HardwareType, SoftwareVersion


class BenchmarkCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    device_name: DeviceName = Field(description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    hardware_type: Optional[HardwareType] = Field(default=None, description="Hardware type")


class BenchmarkPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    framework: TargetFrameworkPayload
    device: TargetDevicePayload
    software_version: Optional[SoftwareVersionPayload] = None
    hardware_type: Optional[HardwareTypePayload] = None
    precision: PrecisionForBenchmarkPayload
    result: BenchmarkResultPayload
    status: str
    is_deleted: bool
    error_detail: Optional[Dict] = None
    input_model_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BenchmarkCreatePayload(BaseModel):
    task_id: str


class BenchmarkCreateResponse(ResponseItem):
    data: BenchmarkCreatePayload


class BenchmarkResponse(ResponseItem):
    data: BenchmarkPayload


class BenchmarksResponse(ResponsePaginationItems):
    data: List[BenchmarkPayload]
    result_count: int
    total_count: int
