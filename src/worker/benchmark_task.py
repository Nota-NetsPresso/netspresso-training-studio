from celery import chain

from src.core.db.session import get_db_session
from src.modules.benchmarker.v2.benchmarker import BenchmarkerV2
from src.worker.celery_app import celery_app

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='benchmark_model')
def benchmark_model(
    self,
    api_key: str,
    input_model_id: str,
    benchmark_task_id: str,
    target_device_name: str,
    target_software_version: str = None,
    target_hardware_type: str = None,
):
    with get_db_session() as db:
        benchmarker = BenchmarkerV2(api_key=api_key)

        task_id = benchmarker.benchmark_model(
            db=db,
            input_model_id=input_model_id,
            benchmark_task_id=benchmark_task_id,
            target_device_name=target_device_name,
            target_software_version=target_software_version,
            target_hardware_type=target_hardware_type,
            wait_until_done=False,
        )

        chain(poll_benchmark_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
        return task_id


@celery_app.task
def poll_benchmark_status(api_key: str, task_id: str):
    with get_db_session() as db:
        benchmarker = BenchmarkerV2(api_key=api_key)
        status_updated = benchmarker.update_benchmark_task_status(db=db, task_id=task_id)

        if not status_updated:
            poll_benchmark_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
