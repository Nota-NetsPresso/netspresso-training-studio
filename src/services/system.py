from typing import List

from gpustat import GPUStatCollection

from src.api.v1.schemas.system import GpuInfoPayload, LibraryInfo
from src.configs.version import BACKEND_VERSION


class SystemService:
    def get_backend_version(self) -> List[LibraryInfo]:
        backend_version = BACKEND_VERSION
        installed_libraries = [
            LibraryInfo(name="backend", version=backend_version),
        ]
        return installed_libraries

    def get_gpus_info(self) -> List[GpuInfoPayload]:
        stats = GPUStatCollection.new_query()

        gpus_info = [
            GpuInfoPayload(
                index=gpu.index,
                uuid=gpu.uuid,
                name=gpu.name,
                temperature_gpu=gpu.temperature,
                fan_speed=gpu.fan_speed,
                utilization_gpu=gpu.utilization,
                utilization_enc=gpu.utilization_enc,
                utilization_dec=gpu.utilization_dec,
                power_draw=gpu.power_draw,
                enforced_power_limit=gpu.power_limit,
                memory_used=gpu.memory_used,
                memory_total=gpu.memory_total,
                processes=list(gpu.processes),
            )
            for gpu in stats.gpus
        ]

        return gpus_info


system_service = SystemService()
