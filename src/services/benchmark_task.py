from pathlib import Path
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.benchmark.benchmark_task import (
    BenchmarkCreate,
    BenchmarkCreatePayload,
    BenchmarkPayload,
    BenchmarkResponse,
    TargetFrameworkPayload,
)
from src.api.v1.schemas.tasks.common.device import (
    BenchmarkResultPayload,
    HardwareTypePayload,
    PrecisionForBenchmarkPayload,
    SoftwareVersionPayload,
    SupportedDeviceForBenchmarkPayload,
    TargetDevicePayload,
)
from src.enums.model import Framework, ModelType
from src.enums.task import TaskStatus
from src.models.base import generate_uuid
from src.models.benchmark import BenchmarkResult, BenchmarkTask
from src.modules.benchmarker.v2.benchmarker import BenchmarkerV2
from src.modules.clients.enums.task import TaskStatusForDisplay
from src.modules.clients.launcher.v2.schemas.common import DeviceInfo
from src.repositories.base import Order, TimeSort
from src.repositories.benchmark import benchmark_task_repository
from src.repositories.conversion import conversion_task_repository
from src.repositories.model import model_repository
from src.services.conversion_task import conversion_task_service
from src.services.project import project_service
from src.worker.benchmark_task import benchmark_model


class BenchmarkTaskService:
    def _create_device_payload(self, device_info: DeviceInfo, input_model_id: str, data_type: str) -> SupportedDeviceForBenchmarkPayload:
        """Create device payload from device information"""
        return SupportedDeviceForBenchmarkPayload(
            input_model_id=input_model_id,
            name=device_info.device_name,
            software_version=(
                device_info.software_versions[0].software_version if device_info.software_versions else None
            ),
            data_type=data_type,
            hardware_type=device_info.hardware_types[0] if device_info.hardware_types else None,
        )

    def _create_benchmark_payload(self, benchmark_task: BenchmarkTask) -> BenchmarkPayload:
        """Create benchmark payload from task"""
        return BenchmarkPayload(
            task_id=benchmark_task.task_id,
            framework=TargetFrameworkPayload(name=benchmark_task.framework),
            device=TargetDevicePayload(name=benchmark_task.device_name),
            software_version=SoftwareVersionPayload(name=benchmark_task.software_version) if benchmark_task.software_version else None,
            hardware_type=HardwareTypePayload(name=benchmark_task.hardware_type) if benchmark_task.hardware_type else None,
            precision=PrecisionForBenchmarkPayload(name=benchmark_task.precision),
            result=BenchmarkResultPayload(**benchmark_task.result.__dict__) if benchmark_task.result else BenchmarkResultPayload(),
            status=benchmark_task.status,
            is_deleted=benchmark_task.is_deleted,
            error_detail=benchmark_task.error_detail,
            input_model_id=benchmark_task.input_model_id,
            created_at=benchmark_task.created_at,
            updated_at=benchmark_task.updated_at,
        )

    def get_supported_devices(
        self, db: Session, model_id: str, api_key: str
    ) -> List[SupportedDeviceForBenchmarkPayload]:
        """Get list of supported devices for benchmark"""
        benchmarker = BenchmarkerV2(api_key=api_key)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if model.type not in [ModelType.TRAINED_MODEL, ModelType.COMPRESSED_MODEL]:
            raise ValueError("Model is not a trained or compressed model")

        unique_conversions = conversion_task_repository.get_unique_completed_tasks(db=db, model_id=model_id)
        logger.info(f"Found {len(unique_conversions)} unique completed conversions")

        unique_device_keys = set()
        unique_devices = []
        checked_combinations = set()

        for conversion_task in unique_conversions:
            framework = conversion_task.framework
            data_type = conversion_task.precision
            input_model_id = conversion_task.model_id
            is_device_specific = framework in [Framework.TENSORRT, Framework.DRPAI]

            framework_data_type = (framework, data_type)
            if not is_device_specific and framework_data_type in checked_combinations:
                continue

            checked_combinations.add(framework_data_type)
            device = conversion_task.device_name
            software_version = conversion_task.software_version if conversion_task.software_version else None

            _supported_options = benchmarker.get_supported_options(
                framework=framework, device=device, software_version=software_version
            )

            for option in _supported_options:
                if option.framework != framework:
                    continue

                for device_info in option.devices:
                    if is_device_specific and device_info.device_name != device:
                        continue

                    if data_type not in device_info.data_types:
                        continue

                    device_key = (device_info.device_name, input_model_id, data_type)
                    if device_key not in unique_device_keys:
                        unique_device_keys.add(device_key)
                        device_payload = self._create_device_payload(device_info, input_model_id, data_type)
                        unique_devices.append(device_payload)

        return unique_devices

    def check_benchmark_task_exists(self, db: Session, benchmark_in: BenchmarkCreate) -> Optional[BenchmarkCreatePayload]:
        logger.info(f"Checking if benchmark task exists for {benchmark_in.input_model_id}")

        # Check if a task with the same options already exists
        existing_tasks = benchmark_task_repository.get_all_by_model_id(
            db=db,
            model_id=benchmark_in.input_model_id
        )

        # Filter tasks by benchmark parameters
        for task in existing_tasks:
            # Check if this task has the same benchmark parameters
            is_same_options = (
                task.device_name == benchmark_in.device_name and
                task.hardware_type == benchmark_in.hardware_type
            )

            # Software version can be None, handle it separately
            is_same_software_version = (
                benchmark_in.software_version is None or
                task.software_version == benchmark_in.software_version
            )

            if is_same_options and is_same_software_version:
                # If task is in NOT_STARTED, IN_PROGRESS, or COMPLETED state, return it
                reusable_states = [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
                if task.status in reusable_states:
                    logger.info(f"Returning existing benchmark task with status {task.status}: {task.task_id}")
                    return BenchmarkCreatePayload(task_id=task.task_id)

                # For STOPPED or ERROR, we'll create a new task below
                logger.info(f"Previous benchmark task ended with status {task.status}, creating new task")
                break

        logger.info(f"No existing benchmark task found for {benchmark_in.input_model_id}")
        return None

    def create_benchmark_task(self, db: Session, benchmark_in: BenchmarkCreate, api_key: str) -> BenchmarkTask:
        try:
            logger.info(f"Creating benchmark task for {benchmark_in.input_model_id}")

            input_model = model_repository.get_by_model_id(db=db, model_id=benchmark_in.input_model_id)
            logger.info(f"Input model: {input_model}")

            conversion_task = conversion_task_repository.get_by_model_id(db=db, model_id=benchmark_in.input_model_id)
            logger.info(f"Conversion task: {conversion_task}")

            benchmark_task = BenchmarkTask(
                framework=conversion_task.framework,
                device_name=benchmark_in.device_name,
                software_version=benchmark_in.software_version,
                precision=conversion_task.precision,
                status=TaskStatus.NOT_STARTED,
                input_model_id=benchmark_in.input_model_id,
                user_id=input_model.user_id,
            )
            benchmark_task.result = BenchmarkResult(task_id=benchmark_task.task_id)
            benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)

            return benchmark_task
        except Exception as e:
            logger.error(f"Error creating benchmark task: {e}")
            raise e

    def start_benchmark_task(
        self,
        benchmark_in: BenchmarkCreate,
        benchmark_task: BenchmarkTask,
        api_key: str,
    ) -> BenchmarkCreatePayload:
        worker_params = {
            "api_key": api_key,
            "input_model_id": benchmark_in.input_model_id,
            "benchmark_task_id": benchmark_task.task_id,
            "target_device_name": benchmark_in.device_name,
            "target_software_version": benchmark_in.software_version,
            "target_hardware_type": benchmark_in.hardware_type,
        }

        _ = benchmark_model.apply_async(
            kwargs=worker_params,
            benchmark_task_id=benchmark_task.task_id,
        )

        return BenchmarkCreatePayload(task_id=benchmark_task.task_id)

    def get_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkResponse:
        """Get benchmark task status and details"""
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

        return self._create_benchmark_payload(benchmark_task)

    def get_benchmark_tasks(self, db: Session, model_id: str, token: str) -> List[BenchmarkPayload]:
        """Get benchmark tasks for a model"""
        conversion_tasks = conversion_task_service.get_conversion_tasks(db, model_id, token)

        if not conversion_tasks:
            return []

        conv_model_ids = [task.model_id for task in conversion_tasks]

        benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
            db=db,
            converted_model_ids=conv_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        return [self._create_benchmark_payload(benchmark_task) for benchmark_task in benchmark_tasks]

    def cancel_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkPayload:
        """Cancel benchmark task"""
        benchmarker = BenchmarkerV2(api_key=api_key)
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

        launcher_status = benchmarker.cancel_benchmark_task(benchmark_task.benchmark_task_uuid)

        if launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
            benchmark_task.status = TaskStatus.STOPPED
            benchmark_task = benchmark_task_repository.save(db, benchmark_task)
        else:
            raise ValueError(f"Failed to cancel benchmark task: {launcher_status.status}")

        return self._create_benchmark_payload(benchmark_task)

    def delete_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkPayload:
        """Delete benchmark task"""
        benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=task_id)
        benchmark_task = benchmark_task_repository.soft_delete(db=db, model=benchmark_task)

        # Delete benchmarked model from model repository
        model = model_repository.get_by_model_id(db=db, model_id=benchmark_task.input_model_id)
        model = model_repository.soft_delete(db=db, model=model)

        return self._create_benchmark_payload(benchmark_task)

    def _get_benchmark_info(self, db: Session, converted_model_ids: List[str]) -> tuple[Optional[str], List[str]]:
        """Get benchmark task information

        Args:
            db: Database session
            converted_model_ids: List of converted model IDs

        Returns:
            tuple: (latest_status, task_ids)
        """
        if not converted_model_ids:
            return None, []

        benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
            db=db,
            converted_model_ids=converted_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        if not benchmark_tasks:
            return None, []

        task_ids = [task.task_id for task in benchmark_tasks]

        benchmark_task = benchmark_task_repository.get_latest_benchmark_task(
            db=db,
            converted_model_ids=converted_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.UPDATED_AT,
        )
        latest_status = benchmark_task.status

        return latest_status, task_ids


benchmark_task_service = BenchmarkTaskService()
