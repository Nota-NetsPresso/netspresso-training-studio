
from enum import Enum


class Origin(str, Enum):
    ROUTER = "router"
    SERVICE = "service"
    REPOSITORY = "repository"
    CLIENT = "client"
    LIBRARY = "library"
    MODULE = "module"
