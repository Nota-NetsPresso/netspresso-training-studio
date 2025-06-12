from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.api.v1.schemas.project import ProjectPayload
from src.enums.sort import Order
from src.exceptions.auth import UnauthorizedUserAccessException
from src.exceptions.project import ProjectAlreadyExistsException, ProjectNameTooLongException
from src.models.project import Project
from src.repositories.project import project_repository
from src.services.user import user_service


class ProjectService:
    def create_project(self, db: Session, project_name: str, token: str) -> Project:
        """Create a new project for the user.

        Args:
            db: Database session
            project_name: Name of the project (max 30 characters)
            token: User authentication token

        Returns:
            Created project

        Raises:
            ProjectNameTooLongException: If project name exceeds 30 characters
            ProjectAlreadyExistsException: If project name already exists for the user
        """
        user_info = user_service.get_user_info(token=token)

        if len(project_name) > 30:
            raise ProjectNameTooLongException(max_length=30, actual_length=len(project_name))

        # Check if project name is already taken by the user
        is_duplicated = self.check_project_duplication(db=db, project_name=project_name, token=token)
        if is_duplicated:
            logger.error(f"Project name '{project_name}' already exists for user {user_info.user_id}")
            raise ProjectAlreadyExistsException(project_name=project_name)

        logger.info(f"Creating project '{project_name}' for user {user_info.user_id}")
        project = Project(
            project_name=project_name,
            user_id=user_info.user_id,
        )
        project = project_repository.save(db=db, model=project)

        project = ProjectPayload.model_validate(project)

        return project

    def check_project_duplication(self, db: Session, project_name: str, token: str) -> bool:
        """Check if project name is already taken by the user.

        Args:
            db: Database session
            project_name: Name of the project to check
            token: User authentication token

        Returns:
            True if project name is already taken, False otherwise
        """
        user_info = user_service.get_user_info(token=token)

        is_duplicated = project_repository.is_project_name_duplicated(
            db=db, project_name=project_name, user_id=user_info.user_id
        )

        return is_duplicated

    def get_projects(
        self,
        *,
        db: Session,
        token: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Order = Order.DESC,
    ) -> List[Project]:
        """Get all projects for the user with pagination.

        Args:
            db: Database session
            token: User authentication token
            start: Starting index for pagination (0-based)
            size: Number of items per page
            order: Sort order (DESC or ASC)

        Returns:
            List of projects
        """
        user_info = user_service.get_user_info(token=token)

        projects = project_repository.get_all_by_user_id(
            db=db, user_id=user_info.user_id, start=start, size=size, order=order
        )
        projects = [ProjectPayload.model_validate(project) for project in projects]

        return projects

    def count_project_by_user_id(self, *, db: Session, token: str) -> int:
        """Count total number of projects for the user.

        Args:
            db: Database session
            token: User authentication token

        Returns:
            Total number of projects
        """
        user_info = user_service.get_user_info(token=token)

        return project_repository.count_by_user_id(db=db, user_id=user_info.user_id)

    def get_project(self, *, db: Session, project_id: str, token: str) -> Project:
        """Get a specific project by ID.

        Args:
            db: Database session
            project_id: ID of the project to retrieve
            token: User authentication token

        Returns:
            Project details

        Raises:
            UnauthorizedUserAccessException: If user is not the owner of the project
        """
        user_info = user_service.get_user_info(token=token)

        project = project_repository.get_by_project_id(db=db, project_id=project_id)
        if user_info.user_id != project.user_id:
            raise UnauthorizedUserAccessException(user_id=user_info.user_id)

        project = ProjectPayload.model_validate(project)

        return project

    def delete_project(self, *, db: Session, project_id: str, token: str) -> Project:
        """Soft delete a project.

        Args:
            db: Database session
            project_id: ID of the project to delete
            token: User authentication token

        Returns:
            Deleted project details

        Raises:
            UnauthorizedUserAccessException: If user is not the owner of the project
        """
        user_info = user_service.get_user_info(token=token)

        project = project_repository.get_by_project_id(db=db, project_id=project_id)
        if user_info.user_id != project.user_id:
            raise UnauthorizedUserAccessException(user_id=user_info.user_id)
        project = project_repository.soft_delete(db=db, model=project)

        project = ProjectPayload.model_validate(project)

        return project


project_service = ProjectService()
