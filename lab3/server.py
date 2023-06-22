from fastapi import FastAPI
import sys


from ProductQualityModel import ProductQualityModel
from router import api_router

model_file_path = "moons_model.pkl"
scaler_file_path = "moons_scaler.pkl"

quality_model = ProductQualityModel(model_file_path, scaler_file_path)
quality_model.load_model()
if quality_model is None:
    print("Server exited")
    sys.exit(1)

api_router.init_model(quality_model)
print(f"scaler.scale_ = {quality_model.scaler.scale_}")
print(f"model.coef_= {quality_model.model.coef_}")

app = FastAPI()
app.include_router(api_router)
print("Server started")
