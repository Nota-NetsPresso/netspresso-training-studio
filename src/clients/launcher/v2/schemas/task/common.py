from dataclasses import dataclass

from src.clients.enums.task import TaskStatusForDisplay


@dataclass
class Device:
    device_brand: str
    device_name: str


@dataclass
class TaskStatusInfo:
    status: TaskStatusForDisplay
