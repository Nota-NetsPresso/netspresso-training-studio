import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from sqlalchemy.orm import Session

from src.enums.conversion import TargetFramework
from src.enums.device import DeviceName, HardwareType, SoftwareVersion
from src.enums.task import TaskStatus
from src.modules.base import NetsPressoBase
from src.modules.clients.auth import TokenHandler
from src.modules.clients.auth.client import auth_client
from src.modules.clients.auth.response_body import UserResponse
from src.modules.clients.compressor.v2.schemas.common import DeviceInfo
from src.modules.clients.enums.task import TaskStatusForDisplay
from src.modules.clients.launcher import launcher_client_v2
from src.modules.clients.launcher.v2.schemas.common import ModelOption
from src.modules.clients.launcher.v2.schemas.task.benchmark.response_body import BenchmarkTask as BenchmarkTaskInfo
from src.modules.enums.credit import ServiceTask
from src.repositories.benchmark import benchmark_task_repository
from src.repositories.model import model_repository
from src.zenko.storage_handler import ObjectStorageHandler

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"

class BenchmarkerV2(NetsPressoBase):
    def __init__(self, api_key: str, verify_ssl: bool = True) -> None:
        """Initialize the Compressor."""
        super().__init__(token_handler=TokenHandler(api_key=api_key, verify_ssl=verify_ssl))
        self.user_info = self.get_user()

    def get_user(self) -> UserResponse:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(
            self.token_handler.tokens.access_token,
            self.token_handler.verify_ssl
        )
        return user_info

    def filter_device_by_version(
        self, device: DeviceInfo, target_software_version: Optional[SoftwareVersion] = None
    ) -> Optional[DeviceInfo]:
        """Filter device by software version.

        Args:
            device: Device information to filter
            target_software_version: Target software version to filter by

        Returns:
            Optional[DeviceInfo]: Filtered device info or None if no matching version
        """
        if target_software_version is None and device.device_name == DeviceName.AWS_T4:
            return device

        filtered_versions = [
            version for version in device.software_versions if version.software_version == target_software_version
        ]

        if filtered_versions:
            device.software_versions = filtered_versions
            return device
        return None

    def filter_devices_by_name_and_version(
        self, devices: List[DeviceInfo], target_device: DeviceName, target_version: Optional[SoftwareVersion] = None
    ) -> List[DeviceInfo]:
        """Filter devices by name and software version.

        Args:
            devices: List of devices to filter
            target_device: Target device name to filter by
            target_version: Target software version to filter by

        Returns:
            List[DeviceInfo]: List of filtered devices
        """
        filtered_devices = [
            self.filter_device_by_version(device, target_version)
            for device in devices
            if device.device_name == target_device
        ]
        return [device for device in filtered_devices if device]

    def get_supported_options(
        self, framework: TargetFramework, device: DeviceName, software_version: Optional[SoftwareVersion] = None
    ) -> List[ModelOption]:
        """Get supported options for given framework, device and software version.

        Args:
            framework: Target framework
            device: Target device name
            software_version: Target software version

        Returns:
            List[ModelOption]: List of supported model options
        """
        self.token_handler.validate_token()

        # Get all options from launcher
        options_response = launcher_client_v2.benchmarker.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=framework,
        )
        supported_options = options_response.data

        # Filter options for specific frameworks
        if framework in [TargetFramework.TENSORRT, TargetFramework.DRPAI]:
            for option in supported_options:
                if option.framework == framework:
                    option.devices = self.filter_devices_by_name_and_version(
                        devices=option.devices, target_device=device, target_version=software_version
                    )

        return supported_options

    def benchmark_model(
        self,
        db: Session,
        input_model_id: str,
        benchmark_task_id: str,
        target_device_name: DeviceName,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ):
        """Benchmark the specified model on the specified device.

        Args:
            input_model_path (str): The file path where the model is located.
            target_device_name (DeviceName): Target device name.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.
            target_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.
            wait_until_done (bool): If True, wait for the benchmark result before returning the function.
                                If False, request the benchmark and return the function immediately.

        Raises:
            e: If an error occurs during the benchmarking of the model.

        Returns:
            str: Benchmark task ID.
        """
        try:
            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_BENCHMARK)

            output_dir = tempfile.mkdtemp(prefix="netspresso_benchmark_")

            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)

            # Download model to temporary directory
            download_dir = Path(output_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)

            local_path = download_dir / Path(input_model.object_path).name

            logger.info(f"Downloading input model from Zenko to temp directory: {local_path}")
            storage_handler.download_file_from_s3(
                bucket_name=BUCKET_NAME,
                local_path=str(local_path),
                object_path=input_model.object_path
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=benchmark_task_id)
            benchmark_task.status = TaskStatus.IN_PROGRESS
            benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.benchmarker.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
            )

            # Upload model_file
            launcher_client_v2.benchmarker.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.benchmarker.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start benchmark task
            benchmark_response = launcher_client_v2.benchmarker.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                data_type=validate_model_response.data.detail.data_type,
                target_device_name=target_device_name,
                hardware_type=target_hardware_type,
                input_layer=validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
            )

            benchmark_task.benchmark_task_uuid = benchmark_response.data.benchmark_task_id
            benchmark_task.result.file_size = validate_model_response.data.file_size_in_mb
            benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    benchmark_response = launcher_client_v2.benchmarker.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=benchmark_response.data.benchmark_task_id,
                    )
                    if benchmark_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                        TaskStatusForDisplay.USER_CANCEL,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if benchmark_response.data.status in [TaskStatusForDisplay.IN_PROGRESS, TaskStatusForDisplay.IN_QUEUE]:
                benchmark_task.status = TaskStatus.IN_PROGRESS
                logger.info(f"Benchmark task was running. Status: {benchmark_response.data.status}")
            elif benchmark_response.data.status == TaskStatusForDisplay.FINISHED:
                self.print_remaining_credit(service_task=ServiceTask.MODEL_BENCHMARK)
                benchmark_task.status = TaskStatus.COMPLETED

                # Save benchmark results
                _benchmark_result = benchmark_response.data.benchmark_result
                benchmark_task.result.processor = _benchmark_result.processor
                benchmark_task.result.memory_footprint_gpu = _benchmark_result.memory_footprint_gpu
                benchmark_task.result.memory_footprint_cpu = _benchmark_result.memory_footprint_cpu
                benchmark_task.result.power_consumption = _benchmark_result.power_consumption
                benchmark_task.result.ram_size = _benchmark_result.ram_size
                benchmark_task.result.latency = _benchmark_result.latency
                benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)

                logger.info("Benchmark task was completed successfully.")
            elif benchmark_response.data.status in [
                TaskStatusForDisplay.ERROR,
                TaskStatusForDisplay.USER_CANCEL,
                TaskStatusForDisplay.TIMEOUT,
            ]:
                benchmark_task.status = TaskStatus.ERROR
                benchmark_task.error_detail = benchmark_response.data.error_log
                benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)
                logger.error(f"Benchmark task was failed. Error: {benchmark_response.data.error_log}")

        except Exception as e:
            benchmark_task.status = TaskStatus.ERROR
            benchmark_task.error_detail = str(e) if e.args else "Unknown error"
            logger.error(f"Exception during benchmark: {e}")
        except KeyboardInterrupt:
            benchmark_task.status = TaskStatus.STOPPED
            logger.info("Benchmark task stopped by user")
        finally:
            benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)

            # Clean up temporary directory (if output directory is a temporary directory)
            if output_dir and os.path.exists(output_dir):
                logger.info(f"Cleaning up temporary files in: {output_dir}")
                try:
                    shutil.rmtree(output_dir)
                    logger.info(f"Successfully removed temporary directory: {output_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")


        return benchmark_task.task_id

    def get_benchmark_task(self, benchmark_task_id: str) -> BenchmarkTaskInfo:
        """Get information about the specified benchmark task using the benchmark task UUID.

        Args:
            benchmark_task_id (str): Benchmark task UUID of the benchmark task.

        Raises:
            e: If an error occurs while retrieving information about the benchmark task.

        Returns:
            BenchmarkTaskInfo: Model benchmark task object.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=benchmark_task_id,
        )
        return response.data

    def cancel_benchmark_task(self, benchmark_task_id: str) -> BenchmarkTaskInfo:
        """Cancel the benchmark task with given benchmark task uuid.

        Args:
            benchmark_task_id (str): Benchmark task UUID of the benchmark task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            BenchmarkTaskInfo: Model benchmark task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=benchmark_task_id,
        )
        return response.data

    def update_benchmark_task_status(self, db: Session, task_id: str) -> bool:
        """Update benchmark task status in DB based on launcher status.

        Args:
            task_id (str): Benchmark task ID to update

        Returns:
            bool: True if status was updated, False if task is still in progress
        """
        try:
            benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=task_id)
            if not benchmark_task:
                logger.error(f"Benchmark task {task_id} not found")
                return True

            status_updated = False

            if not benchmark_task.benchmark_task_uuid:
                benchmark_task.status = TaskStatus.COMPLETED
                benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)
                logger.info(f"Benchmark task {task_id} status updated to {benchmark_task.status}")
                return True

            launcher_status = self.get_benchmark_task(benchmark_task.benchmark_task_uuid)

            if launcher_status.status == TaskStatusForDisplay.FINISHED:
                benchmark_task.status = TaskStatus.COMPLETED
                status_updated = True

                if launcher_status.benchmark_result:
                    benchmark_task.result.processor = launcher_status.benchmark_result.processor
                    benchmark_task.result.memory_footprint_gpu = launcher_status.benchmark_result.memory_footprint_gpu
                    benchmark_task.result.memory_footprint_cpu = launcher_status.benchmark_result.memory_footprint_cpu
                    benchmark_task.result.power_consumption = launcher_status.benchmark_result.power_consumption
                    benchmark_task.result.ram_size = launcher_status.benchmark_result.ram_size
                    benchmark_task.result.latency = launcher_status.benchmark_result.latency
                    benchmark_task = benchmark_task_repository.update(db=db, model=benchmark_task)

            elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
                benchmark_task.status = TaskStatus.ERROR
                benchmark_task.error_detail = launcher_status.error_log
                status_updated = True

            elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
                benchmark_task.status = TaskStatus.STOPPED
                status_updated = True

            if status_updated:
                benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)
                logger.info(f"Benchmark task {task_id} status updated to {benchmark_task.status}")

            return status_updated

        except Exception as e:
            logger.error(f"Error updating benchmark task status: {e}")

            return True
