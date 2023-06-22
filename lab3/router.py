from PartMeasure import PartMeasure
from RouterWithModel import RouterWithModel


api_router = RouterWithModel()
print("Router is inited")

@api_router.get("/")
async def root():
    return {"message": "Part quality conformity check"}


@api_router.post("/check_quality/")
def check_validity(part: PartMeasure):
    """Part quality conformity check by its measurements
    - **length**: part length
    - **width**: part width
    """
    if api_router.quality_model is None:
        return None

    result = api_router.quality_model.predict(part.length, part.width)
    return result
