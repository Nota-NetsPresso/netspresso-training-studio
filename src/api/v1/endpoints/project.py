from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_token
from src.api.v1.schemas.base import Order
from src.api.v1.schemas.project import (
    ProjectCreate,
    ProjectDuplicationCheckResponse,
    ProjectDuplicationStatus,
    ProjectResponse,
    ProjectsResponse,
)
from src.api.v1.schemas.user import Token
from src.core.db.session import get_db
from src.services.project import project_service

router = APIRouter()


@router.post("", response_model=ProjectResponse)
def create_project(
    *,
    request_body: ProjectCreate,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ProjectResponse:
    project = project_service.create_project(
        db=db,
        project_name=request_body.project_name,
        token=token.access_token,
    )

    return ProjectResponse(data=project)


@router.post("/duplicate", response_model=ProjectDuplicationCheckResponse)
def check_project_duplication(
    *,
    request_body: ProjectCreate,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ProjectDuplicationCheckResponse:
    is_duplicated = project_service.check_project_duplication(
        db=db,
        project_name=request_body.project_name,
        token=token.access_token,
    )

    duplication_status = ProjectDuplicationStatus(is_duplicated=is_duplicated)

    return ProjectDuplicationCheckResponse(data=duplication_status)


@router.get("", response_model=ProjectsResponse)
def get_projects(
    *,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
    start: Optional[int] = 0,
    size: Optional[int] = 10,
    order: Order = Order.DESC,
) -> ProjectsResponse:
    projects = project_service.get_projects(
        db=db,
        start=start,
        size=size,
        order=order,
        token=token.access_token,
    )
    total_count = project_service.count_project_by_user_id(
        db=db,
        token=token.access_token,
    )

    return ProjectsResponse(data=projects, result_count=len(projects), total_count=total_count)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    *,
    project_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ProjectResponse:
    project = project_service.get_project(
        db=db,
        project_id=project_id,
        token=token.access_token,
    )

    return ProjectResponse(data=project)


@router.delete("/{project_id}", response_model=ProjectResponse)
def delete_project(
    *,
    project_id: str,
    db: Session = Depends(get_db),
    token: Token = Depends(get_token),
) -> ProjectResponse:
    project = project_service.delete_project(
        db=db,
        project_id=project_id,
        token=token.access_token,
    )

    return ProjectResponse(data=project)
