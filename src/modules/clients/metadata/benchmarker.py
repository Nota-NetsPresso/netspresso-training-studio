from dataclasses import dataclass, field
from typing import List

from src.enums.device import DeviceName, HardwareType, SoftwareVersion
from src.enums.model import DataType, Framework
from src.enums.task import TaskType
from src.modules.clients.metadata.common import BaseMetadata


@dataclass
class BenchmarkTaskInfo:
    benchmark_task_uuid: str = ""
    framework: Framework = ""
    display_framework: str = ""
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    software_version: SoftwareVersion = ""
    display_software_version: str = ""
    data_type: DataType = ""
    hardware_type: HardwareType = ""


@dataclass
class BenchmarkResult:
    memory_footprint: int = 0
    memory_footprint_gpu: int = 0
    memory_footprint_cpu: int = 0
    power_consumption: int = 0
    ram_size: int = 0
    latency: int = 0
    file_size: int = 0


@dataclass
class BenchmarkEnvironment:
    model_framework: str = ""
    system: str = ""
    machine: str = ""
    cpu: str = ""
    gpu: str = ""
    library: List[str] = field(default_factory=list)


@dataclass
class BenchmarkerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.BENCHMARK
    input_model_path: str = ""
    benchmark_task_info: BenchmarkTaskInfo = field(default_factory=BenchmarkTaskInfo)
    benchmark_result: BenchmarkResult = field(default_factory=BenchmarkResult)
