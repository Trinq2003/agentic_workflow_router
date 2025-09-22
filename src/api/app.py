from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from strategy.nlp_strategy import WorkerLabelingNLPStrategy

import logging, os

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

class QueryRequest(BaseModel):
    query: str


class LabelsResponse(BaseModel):
    labels: List[str]


app = FastAPI(title="NetMind Workflow API", version="0.1.0")


# Lazily initialize strategy to avoid expensive imports at startup if not needed
strategy_instance: WorkerLabelingNLPStrategy | None = None


def get_strategy() -> WorkerLabelingNLPStrategy:
    global strategy_instance
    if strategy_instance is None:
        strategy_instance = WorkerLabelingNLPStrategy()
    return strategy_instance


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def available_labels() -> LabelsResponse:
    strategy = get_strategy()
    return LabelsResponse(labels=strategy.workers)


@app.post("/label", response_model=LabelsResponse)
def label_query(payload: QueryRequest) -> LabelsResponse:
    if not payload.query or not isinstance(payload.query, str):
        raise HTTPException(status_code=400, detail="Field 'query' must be a non-empty string")

    strategy = get_strategy()
    try:
        arr = strategy.forward(payload.query)
        vector = np.squeeze(arr)[0:].tolist() if hasattr(arr, "tolist") else list(arr)
        selected = [strategy.workers[i] for i, v in enumerate(vector) if int(v) == 1]
        return LabelsResponse(labels=selected)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}")


