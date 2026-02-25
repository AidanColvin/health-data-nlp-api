from fastapi import APIRouter
from src.api.schemas import PredictRequest, PredictResponse, Entities
from src.model.inference import predict

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest) -> PredictResponse:
    specialty, entities, score = predict(req.transcription)
    return PredictResponse(
        specialty=specialty,
        entities=Entities(**entities),
        confidence_score=score,
    )
