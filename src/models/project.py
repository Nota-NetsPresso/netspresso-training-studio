from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from src.enums.model import ModelType
from src.models.base import BaseModel, generate_uuid


class Project(BaseModel):
    __tablename__ = "project"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    project_id = Column(
        String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="project")
    )
    project_name = Column(String(30), nullable=False, unique=True)
    user_id = Column(String(36), nullable=False)

    # Relationship to Model
    models = relationship(
        "Model",
        back_populates="project",
        cascade="all",
        lazy="joined",
    )

    # Property to get model IDs
    @hybrid_property
    def model_ids(self):
        return [
            model.model_id
            for model in self.models
            if model.type in [ModelType.TRAINED_MODEL, ModelType.COMPRESSED_MODEL] and not model.is_deleted
        ]
