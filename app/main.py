import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import CustomerIn, PredictionOut

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/churn_pipeline.pkl"))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn):
    df = pd.DataFrame([customer.model_dump()])
    proba = float(state["pipeline"].predict_proba(df)[0, 1])
    return PredictionOut(
        churn_probability=round(proba, 4),
        churn_prediction=int(proba >= THRESHOLD),
        threshold=THRESHOLD,
    )
