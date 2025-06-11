from enum import Enum


class Order(str, Enum):
    DESC = "desc"
    ASC = "asc"


class TimeSort(str, Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
