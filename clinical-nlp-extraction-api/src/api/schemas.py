from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    transcription: str = Field(..., min_length=1)

class Entities(BaseModel):
    symptoms: List[str] = []
    negated_symptoms: List[str] = []

class PredictResponse(BaseModel):
    specialty: str
    entities: Entities
    confidence_score: float
