from celery import chain
from sqlalchemy.orm import Session

from src.core.db.session import get_db_session
from src.enums.model import DataType
from src.modules.converter.v2.converter import ConverterV2
from src.worker.celery_app import celery_app

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='convert_model')
def convert_model(
    self,
    api_key: str,
    input_model_id: str,
    conversion_task_id: str,
    target_framework: str,
    target_device_name: str,
    target_data_type: str = DataType.FP16,
    target_software_version: str = None,
    dataset_path: str = None,
):
    with get_db_session() as db:
        converter = ConverterV2(api_key=api_key)

        task_id = converter.convert_model(
            db=db,
            input_model_id=input_model_id,
            conversion_task_id=conversion_task_id,
            target_framework=target_framework,
            target_device_name=target_device_name,
            target_data_type=target_data_type,
            target_software_version=target_software_version,
            dataset_path=dataset_path,
            wait_until_done=False,
        )

        chain(poll_conversion_status.s(db=db, api_key=api_key, task_id=task_id).set(countdown=POLLING_INTERVAL))()
        return task_id


@celery_app.task
def poll_conversion_status(db: Session, api_key: str, task_id: str):
    converter = ConverterV2(api_key=api_key)
    status_updated = converter.update_conversion_task_status(db=db, task_id=task_id)

    if not status_updated:
        poll_conversion_status.apply_async(args=[db, api_key, task_id], countdown=POLLING_INTERVAL)
