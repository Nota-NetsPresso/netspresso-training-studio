from pathlib import Path
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.tasks.common.device import (
    HardwareTypePayload,
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
    TargetDevicePayload,
)
from src.api.v1.schemas.tasks.conversion.conversion_task import (
    ConversionCreate,
    ConversionCreatePayload,
    ConversionPayload,
    TargetFrameworkPayload,
)
from src.enums.conversion import SourceFramework, TargetFramework
from src.enums.device import DeviceName, SoftwareVersion
from src.enums.model import DataType, ModelType
from src.enums.task import TaskStatus
from src.models.base import generate_uuid
from src.models.conversion import ConversionTask
from src.models.model import Model
from src.modules.clients.enums.task import TaskStatusForDisplay
from src.modules.clients.launcher.v2.schemas.common import DeviceInfo
from src.modules.converter.v2.converter import ConverterV2
from src.repositories.base import Order, TimeSort
from src.repositories.conversion import conversion_task_repository
from src.repositories.model import model_repository
from src.utils.file import FileHandler
from src.worker.conversion_task import convert_model


class ConversionTaskService:
    def get_supported_devices(
        self, db: Session, framework: SourceFramework, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (SourceFramework): Framework to get supported devices for
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        converter = ConverterV2(api_key=api_key)
        supported_options = converter.get_supported_options(framework=framework)

        return [self._create_supported_device_response(option) for option in supported_options]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        return SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

    def _create_device_payload(self, device: DeviceInfo) -> SupportedDevicePayload:
        """Create SupportedDevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            SupportedDevicePayload: Payload containing device information
        """
        return SupportedDevicePayload(
            name=device.device_name,
            software_versions=[
                SoftwareVersionPayload(name=version.software_version) for version in device.software_versions
            ],
            precisions=[PrecisionForConversionPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def _generate_model_name(
        self,
        input_model_name: str,
        framework: TargetFramework,
        device_name: DeviceName,
        data_type: DataType,
        software_version: Optional[SoftwareVersion],
    ) -> str:
        # Generate model name with safe enum value handling
        model_name_parts = [
            input_model_name,
            framework.value,
            device_name.value,
        ]
        if software_version:  # Add only if not None
            model_name_parts.append(software_version.value)

        model_name_parts.append(data_type.value)
        model_name = "_".join(map(str, model_name_parts))
        logger.info(f"Model name: {model_name}")

        return model_name

    def create_converted_model(self, db: Session, model: Model, framework: TargetFramework, device_name: DeviceName, data_type: DataType, software_version: Optional[SoftwareVersion]) -> Model:
        model_id = generate_uuid(entity="model")
        extension = FileHandler.get_extension(framework=framework)
        base_object_path = f"{model.user_id}/{model.project_id}/{model_id}/model{extension}"
        model_name = self._generate_model_name(
            input_model_name=model.name,
            framework=framework,
            device_name=device_name,
            data_type=data_type,
            software_version=software_version,
        )
        model_obj = Model(
            model_id=model_id,
            name=model_name,
            type=ModelType.CONVERTED_MODEL,
            project_id=model.project_id,
            user_id=model.user_id,
            object_path=base_object_path
        )
        model_obj = model_repository.save(db=db, model=model_obj)

        return model_obj

    def check_conversion_task_exists(self, db: Session, conversion_in: ConversionCreate) -> Optional[ConversionCreatePayload]:
        # Check if a task with the same options already exists
        existing_tasks = conversion_task_repository.get_all_by_model_id(
            db=db,
            model_id=conversion_in.input_model_id
        )

        # Filter tasks by conversion parameters
        for task in existing_tasks:
            # Check if this task has the same conversion parameters
            is_same_options = (
                task.framework == conversion_in.framework and
                task.device_name == conversion_in.device_name and
                task.precision == conversion_in.precision
            )

            # Software version can be None, handle it separately
            is_same_software_version = (
                conversion_in.software_version is None or
                task.software_version == conversion_in.software_version
            )

            if is_same_options and is_same_software_version:
                # If task is in NOT_STARTED, IN_PROGRESS, or COMPLETED state, return it
                reusable_states = [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
                if task.status in reusable_states:
                    logger.info(f"Returning existing conversion task with status {task.status}: {task.task_id}")
                    return ConversionCreatePayload(task_id=task.task_id)

                # For STOPPED or ERROR, we'll create a new task below
                logger.info(f"Previous conversion task ended with status {task.status}, creating new task")
                break

        return None

    def create_conversion_task(self, db: Session, conversion_in: ConversionCreate, api_key: str) -> ConversionTask:
        input_model = model_repository.get_by_model_id(db=db, model_id=conversion_in.input_model_id)

        converted_model = self.create_converted_model(
            db=db,
            model=input_model,
            framework=conversion_in.framework,
            device_name=conversion_in.device_name,
            data_type=conversion_in.precision,
            software_version=conversion_in.software_version,
        )

        conversion_task = ConversionTask(
            framework=conversion_in.framework,
            device_name=conversion_in.device_name,
            software_version=conversion_in.software_version,
            precision=conversion_in.precision,
            status=TaskStatus.NOT_STARTED,
            input_model_id=conversion_in.input_model_id,
            model_id=converted_model.model_id,
            user_id=input_model.user_id,
        )
        conversion_task = conversion_task_repository.save(db=db, model=conversion_task)

        return conversion_task

    def start_conversion_task(
        self,
        conversion_in: ConversionCreate,
        conversion_task: ConversionTask,
        api_key: str,
    ) -> ConversionCreatePayload:
        worker_params = {
            "api_key": api_key,
            "input_model_id": conversion_in.input_model_id,
            "conversion_task_id": conversion_task.task_id,
            "target_framework": conversion_in.framework,
            "target_device_name": conversion_in.device_name,
            "target_data_type": conversion_in.precision,
            "target_software_version": conversion_in.software_version,
        }

        _ = convert_model.apply_async(
            kwargs=worker_params,
            conversion_task_id=conversion_task.task_id,
        )

        return ConversionCreatePayload(task_id=conversion_task.task_id)

    def get_conversion_task(self, db: Session, task_id: str, api_key: str) -> ConversionPayload:
        conversion_task = conversion_task_repository.get_by_task_id(db, task_id)

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload

    def cancel_conversion_task(self, db: Session, task_id: str, api_key: str):
        converter = ConverterV2(api_key=api_key)
        conversion_task = conversion_task_repository.get_by_task_id(db, task_id)
        convert_task = converter.cancel_conversion_task(conversion_task.convert_task_uuid)

        if convert_task.status == TaskStatusForDisplay.USER_CANCEL:
            conversion_task.status = TaskStatus.STOPPED
            conversion_task = conversion_task_repository.save(db, conversion_task)
        else:
            raise ValueError(f"Failed to cancel conversion task: {convert_task.status}")

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload

    def _create_conversion_payload(self, conversion_task: ConversionTask) -> ConversionPayload:
        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        return ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

    def get_conversion_tasks(self, db: Session, model_id: str, token: str) -> List[ConversionPayload]:
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db, model_id=model_id, order=Order.DESC, time_sort=TimeSort.CREATED_AT,
        )

        return [self._create_conversion_payload(task) for task in conversion_tasks]

    def delete_conversion_task(self, db: Session, task_id: str, api_key: str) -> ConversionPayload:
        conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=task_id)
        conversion_task = conversion_task_repository.soft_delete(db=db, model=conversion_task)

        # Delete converted model from model repository
        model = model_repository.get_by_model_id(db=db, model_id=conversion_task.model_id)
        model = model_repository.soft_delete(db=db, model=model)

        return self._create_conversion_payload(conversion_task)

    def _get_conversion_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str], List[str]]:
        # Get all conversion tasks sorted by creation time (newest first)
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db, model_id=model_id, order=Order.DESC, time_sort=TimeSort.CREATED_AT,
        )

        task_ids = []
        model_ids = []

        for task in conversion_tasks:
            task_ids.append(task.task_id)
            model_ids.append(task.model_id)

        latest_status = conversion_tasks[0].status if conversion_tasks else None

        return latest_status, task_ids, model_ids


conversion_task_service = ConversionTaskService()
