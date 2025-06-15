import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.modules.clients.launcher.v2.schemas import InputLayer
from src.enums.device import DeviceName, SoftwareVersion
from src.enums.model import DataType, Framework


@dataclass
class RequestConvert:
    input_model_id: str
    target_framework: Framework
    target_device_name: DeviceName
    data_type: Optional[DataType] = None
    input_layer: Optional[InputLayer] = field(default_factory=InputLayer)
    software_version: Optional[SoftwareVersion] = ""

    def __post_init__(self):
        if self.input_layer:
            self.input_layer = json.dumps(asdict(self.input_layer))
