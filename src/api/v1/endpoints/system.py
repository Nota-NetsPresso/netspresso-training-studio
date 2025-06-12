from fastapi import APIRouter

from src.api.v1.schemas.system import GpusInfoResponse, LibraryInfo, ServerInfoPayload, ServerInfoResponse
from src.configs.version import BACKEND_VERSION
from src.services.system import system_service

router = APIRouter()


@router.get("/server-info", response_model=ServerInfoResponse)
def get_server_info() -> ServerInfoResponse:
    installed_libraries = system_service.get_backend_version()

    server_info = ServerInfoPayload(installed_libraries=installed_libraries)

    return ServerInfoResponse(data=server_info)


@router.get("/gpus-info", response_model=GpusInfoResponse)
def get_gpus_info() -> GpusInfoResponse:

    gpus_info = system_service.get_gpus_info()

    return GpusInfoResponse(data=gpus_info)
