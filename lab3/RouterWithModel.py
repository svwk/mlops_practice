from fastapi import APIRouter
from ProductQualityModel import ProductQualityModel


class RouterWithModel(APIRouter):
    def __init__(self):
        super().__init__()
        self.quality_model = None

    def init_model(self, model: ProductQualityModel):
        self.quality_model = model
