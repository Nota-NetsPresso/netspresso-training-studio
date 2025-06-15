from dataclasses import dataclass
from typing import Optional

from src.clients.launcher.v2.schemas import InputLayer
from src.enums.device import DeviceName, HardwareType, SoftwareVersion
from src.enums.model import DataType


@dataclass
class RequestBenchmark:
    input_model_id: str
    target_device_name: DeviceName
    data_type: Optional[DataType] = None
    software_version: Optional[SoftwareVersion] = ""
    hardware_type: Optional[HardwareType] = None
    input_layer: Optional[InputLayer] = None
