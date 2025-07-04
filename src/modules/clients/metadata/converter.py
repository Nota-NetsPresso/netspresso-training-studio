from dataclasses import dataclass, field
from typing import List

from src.enums.device import DeviceName, SoftwareVersion
from src.enums.model import DataType, Framework
from src.enums.task import RetrievalTaskType
from src.modules.clients.metadata.common import AvailableOption, BaseMetadata, ModelInfo


@dataclass
class ConvertInfo:
    convert_task_uuid: str = ""
    framework: Framework = ""
    display_framework: str = ""
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    data_type: DataType = ""
    software_version: SoftwareVersion = ""
    display_software_version: str = ""
    model_file_name: str = ""
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class ConverterMetadata(BaseMetadata):
    task_type: RetrievalTaskType = RetrievalTaskType.CONVERT
    input_model_path: str = ""
    converted_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    convert_task_info: ConvertInfo = field(default_factory=ConvertInfo)
    available_options: List[AvailableOption] = field(default_factory=list)
