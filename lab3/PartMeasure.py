from pydantic import BaseModel


class PartMeasure(BaseModel):
    length: float
    width: float
