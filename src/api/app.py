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
    votes: List[float] | None = None


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
        result = strategy.forward(payload.query)

        # If strategy returns dict with labels and votes (new behavior)
        if isinstance(result, dict) and "labels" in result and "votes" in result:
            labels = result.get("labels", [])
            votes = result.get("votes", [])
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise HTTPException(status_code=500, detail="Strategy returned invalid labels format")
            try:
                votes = [float(v) for v in votes]
            except Exception:
                raise HTTPException(status_code=500, detail="Strategy returned invalid votes format")
            return LabelsResponse(labels=labels, votes=votes)

        # If strategy already returns a list of worker names
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return LabelsResponse(labels=result, votes=None)

        # Backward compatibility: handle numeric vote vectors
        vector_like = np.squeeze(result)
        vector = vector_like.tolist() if hasattr(vector_like, "tolist") else list(vector_like)

        # Flatten any nested single-dimension lists
        if len(vector) > 0 and isinstance(vector[0], (list, tuple)):
            # e.g., [[0,1,0,...]] -> [0,1,0,...]
            vector = list(vector[0])

        # Convert all values to float safely
        try:
            numeric_vector = [float(v) for v in vector]
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to interpret model output as votes")

        # Determine selected labels
        unique_vals = set(numeric_vector)
        selected_indices: list[int]
        if unique_vals.issubset({0.0, 1.0}):
            selected_indices = [i for i, v in enumerate(numeric_vector) if v == 1.0]
        else:
            max_val = max(numeric_vector) if numeric_vector else 0.0
            selected_indices = [i for i, v in enumerate(numeric_vector) if v == max_val and max_val > 0.0]

        selected = [strategy.workers[i] for i in selected_indices if i < len(strategy.workers)]
        selected_votes = [numeric_vector[i] for i in selected_indices if i < len(strategy.workers)]
        return LabelsResponse(labels=selected, votes=selected_votes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}")


