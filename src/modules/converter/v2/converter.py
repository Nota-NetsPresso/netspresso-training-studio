import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union
from urllib import request

from loguru import logger
from sqlalchemy.orm import Session

from src.configs.settings import settings
from src.enums.conversion import SourceFramework, TargetFramework
from src.enums.device import DeviceName, SoftwareVersion
from src.enums.model import DataType, ModelType
from src.enums.task import TaskStatus
from src.modules.base import NetsPressoBase
from src.modules.clients.auth import TokenHandler, auth_client
from src.modules.clients.auth.response_body import UserResponse
from src.modules.clients.enums.task import TaskStatusForDisplay
from src.modules.clients.launcher import launcher_client_v2
from src.modules.clients.launcher.v2.schemas.common import ModelOption
from src.modules.clients.launcher.v2.schemas.task.convert.response_body import ConvertTask
from src.modules.enums.credit import ServiceTask
from src.repositories.compression import compression_task_repository
from src.repositories.conversion import conversion_task_repository
from src.repositories.model import model_repository
from src.zenko.storage_handler import ObjectStorageHandler

storage_handler = ObjectStorageHandler()


class ConverterV2(NetsPressoBase):
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

    def get_supported_options(self, framework: SourceFramework) -> List[ModelOption]:
        """Get supported options for the specified framework.

        Args:
            framework: Source framework to get options for

        Returns:
            List of supported model options
        """
        self.token_handler.validate_token()

        options_response = launcher_client_v2.converter.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=framework,
        )
        supported_options = options_response.data

        # Filter out DLC framework (will be removed when DLC is supported)
        supported_options = [
            supported_option
            for supported_option in supported_options
            if supported_option.framework != "dlc"
        ]

        return supported_options

    def _download_converted_model(self, convert_task: ConvertTask, local_path: str) -> None:
        """Download the converted model.

        Args:
            convert_task: Conversion task containing the model
            local_path: Path to save the downloaded model

        Raises:
            Exception: If download fails
        """
        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.converter.download_model_file(
                convert_task_uuid=convert_task.convert_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e

    def convert_model(
        self,
        db: Session,
        input_model_id: str,
        conversion_task_id: str,
        target_framework: Union[str, TargetFramework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> str:
        """Convert a model to the specified framework.

        Args:
            input_model_path: The file path where the model is located.
            output_dir: The local folder path to save the converted model.
            target_framework: The target framework name.
            target_device_name: Target device name.
            target_data_type: Data type of the model. Default is DataType.FP16.
            target_software_version: Target software version.
                Required if target_device_name is one of the Jetson devices.
            input_layer: Target input shape for conversion (e.g., dynamic batch to static batch).
            dataset_path: Path to the dataset. Useful for certain conversions.
            wait_until_done: If True, wait for the conversion result before returning.
                If False, request the conversion and return the function immediately.
            input_model_id: Model ID to convert (alternative to input_model_path)
            project_id: Project ID for the model (required when using input_model_path)

        Raises:
            ValueError: If neither input_model_id nor input_model_path is provided, or if
                       input_model_path is provided without project_id

        Returns:
            str: Conversion task ID.
        """
        try:
            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_CONVERT)

            output_dir = tempfile.mkdtemp(prefix="netspresso_conversion_")

            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)

            # Download model to temporary directory
            download_dir = Path(output_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)

            if input_model.type == ModelType.COMPRESSED_MODEL:
                remote_model_path = Path(input_model.object_path).parent / "model.onnx"
            else:
                remote_model_path = Path(input_model.object_path) / "model.onnx"

            local_path = download_dir / "model.onnx"

            logger.info(f"Downloading input model from Zenko: {remote_model_path}")
            storage_handler.download_file_from_s3(
                bucket_name=settings.MODEL_BUCKET_NAME,
                local_path=str(local_path),
                object_path=str(remote_model_path)
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=conversion_task_id)
            conversion_task.status = TaskStatus.IN_PROGRESS
            conversion_task = conversion_task_repository.update(db=db, model=conversion_task)

            if target_data_type == DataType.INT8:
                if input_model.type == ModelType.COMPRESSED_MODEL:
                    compression_task = compression_task_repository.get_by_model_id(db=db, model_id=input_model.model_id)
                    original_input_model = model_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
                    remote_calibration_dataset_path = Path(original_input_model.object_path) / "calibration_dataset.npy"
                else:
                    remote_calibration_dataset_path = Path(input_model.object_path) / "calibration_dataset.npy"
                local_calibration_dataset_path = download_dir / "calibration_dataset.npy"

                logger.info(f"Downloading calibration dataset from Zenko: {remote_calibration_dataset_path}")
                storage_handler.download_file_from_s3(
                    bucket_name=settings.MODEL_BUCKET_NAME,
                    local_path=str(local_calibration_dataset_path),
                    object_path=str(remote_calibration_dataset_path)
                )
                logger.info(f"Downloaded calibration dataset from Zenko: {local_calibration_dataset_path}")

                dataset_path = local_calibration_dataset_path.as_posix()

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.converter.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
            )

            # Upload model_file
            launcher_client_v2.converter.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.converter.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=local_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Get input layer information
            if not validate_model_response.data.detail.input_layers:
                raise ValueError("Input layer information not found")
            actual_input_layer = validate_model_response.data.detail.input_layers[0]

            # Start convert task
            convert_response = launcher_client_v2.converter.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                target_device_name=target_device_name,
                target_framework=target_framework,
                data_type=target_data_type,
                input_layer=actual_input_layer,
                software_version=target_software_version,
                dataset_path=dataset_path,
            )
            logger.info(f"Convert response: {convert_response.data}")

            conversion_task.convert_task_uuid = convert_response.data.convert_task_id
            conversion_task = conversion_task_repository.update(db=db, model=conversion_task)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    convert_response = launcher_client_v2.converter.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=convert_response.data.convert_task_id,
                    )
                    if convert_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                        TaskStatusForDisplay.USER_CANCEL,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if convert_response.data.status in [TaskStatusForDisplay.IN_PROGRESS, TaskStatusForDisplay.IN_QUEUE]:
                conversion_task.status = TaskStatus.IN_PROGRESS
                logger.info(f"Conversion task running. Status: {convert_response.data.status}")
            elif convert_response.data.status == TaskStatusForDisplay.FINISHED:
                download_dir = Path(conversion_task.model.object_path).parent
                download_dir.mkdir(parents=True, exist_ok=True)
                self._download_converted_model(
                    convert_task=convert_response.data,
                    local_path=conversion_task.model.object_path,
                )
                storage_handler.upload_file_to_s3(
                    bucket_name=settings.MODEL_BUCKET_NAME,
                    local_path=conversion_task.model.object_path,
                    object_path=conversion_task.model.object_path
                )
                logger.info(f"Uploaded Converted Model file to Zenko: {conversion_task.model.object_path}")
                self.print_remaining_credit(service_task=ServiceTask.MODEL_CONVERT)
                conversion_task.status = TaskStatus.COMPLETED
                logger.info("Conversion task completed successfully.")
            elif convert_response.data.status in [
                TaskStatusForDisplay.ERROR,
                TaskStatusForDisplay.USER_CANCEL,
                TaskStatusForDisplay.TIMEOUT,
            ]:
                conversion_task.status = TaskStatus.ERROR
                conversion_task.error_detail = convert_response.data.error_log
                conversion_task = conversion_task_repository.update(db=db, model=conversion_task)
                logger.error(f"Conversion task failed. Error: {convert_response.data.error_log}")

        except Exception as e:
            conversion_task.status = TaskStatus.ERROR
            conversion_task.error_detail = str(e) if e.args else "Unknown error"
            logger.error(f"Exception during conversion: {e}")
        except KeyboardInterrupt:
            conversion_task.status = TaskStatus.STOPPED
            logger.info("Conversion task stopped by user")
        finally:
            conversion_task = conversion_task_repository.update(db=db, model=conversion_task)

            # Clean up temporary directory (if output directory is a temporary directory)
            if output_dir and os.path.exists(output_dir):
                logger.info(f"Cleaning up temporary files in: {output_dir}")
                try:
                    shutil.rmtree(output_dir)
                    logger.info(f"Successfully removed temporary directory: {output_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")

        return conversion_task.task_id

    def get_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Get the conversion task information with given conversion task uuid.

        Args:
            conversion_task_id: Convert task UUID of the convert task.

        Raises:
            Exception: If an error occurs during retrieval.

        Returns:
            ConversionTask: Model conversion task data.
        """
        self.token_handler.validate_token()

        response = launcher_client_v2.converter.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data

    def cancel_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Cancel the conversion task with given conversion task uuid.

        Args:
            conversion_task_id: Convert task UUID of the convert task.

        Raises:
            Exception: If an error occurs during task cancellation.

        Returns:
            ConversionTask: Model conversion task data.
        """
        self.token_handler.validate_token()

        response = launcher_client_v2.converter.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data

    def update_conversion_task_status(self, db: Session, task_id: str) -> bool:
        """Update conversion task status in DB based on launcher status.

        Args:
            task_id (str): Conversion task ID to update

        Returns:
            bool: True if status was updated, False if task is still in progress
        """
        try:
            conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=task_id)
            if not conversion_task:
                logger.error(f"Conversion task {task_id} not found")
                return True

            if not conversion_task.convert_task_uuid:
                conversion_task.status = TaskStatus.COMPLETED
                conversion_task_repository.save(db, conversion_task)
                logger.info(f"Conversion task {task_id} status updated to {conversion_task.status}")
                return True

            launcher_status = self.get_conversion_task(conversion_task.convert_task_uuid)
            status_updated = False

            if launcher_status.status == TaskStatusForDisplay.FINISHED:
                conversion_task.status = TaskStatus.COMPLETED
                status_updated = True
                model = model_repository.get_by_model_id(db=db, model_id=conversion_task.model_id)
                download_dir = Path(model.object_path).parent
                download_dir.mkdir(parents=True, exist_ok=True)
                self._download_converted_model(
                    convert_task=launcher_status,
                    local_path=model.object_path,
                )
                logger.info(f"Downloaded model to {model.object_path}")
                storage_handler.upload_file_to_s3(
                    bucket_name=settings.MODEL_BUCKET_NAME,
                    local_path=model.object_path,
                    object_path=model.object_path
                )
                logger.info(f"Uploaded Converted Model file to Zenko: {model.object_path}")

            elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
                conversion_task.status = TaskStatus.ERROR
                conversion_task.error_detail = launcher_status.error_log
                status_updated = True

            elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
                conversion_task.status = TaskStatus.STOPPED
                status_updated = True

            if status_updated:
                conversion_task = conversion_task_repository.save(db=db, model=conversion_task)
                logger.info(f"Conversion task {task_id} status updated to {conversion_task.status}")

            return status_updated

        except Exception as e:
            logger.error(f"Error updating conversion task status: {e}")

            return True
