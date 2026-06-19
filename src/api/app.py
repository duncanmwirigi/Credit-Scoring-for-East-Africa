from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchScoreRequest,
    FeatureContributionResponse,
    HealthResponse,
    ModelInfoResponse,
    PolicyResponse,
    ScoreRequest,
    ScoreResponse,
    ShapExplanationResponse,
)
from src.config import AppConfig, load_config
from src.ml.scorer import CreditScorer

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    config: AppConfig
    scorer: CreditScorer
    model_version: str
    model_file: str


def _load_state() -> AppState:
    config = load_config()
    scorer = CreditScorer.from_latest(config)
    latest_file = config.model_dir / "latest_model.txt"
    model_file = latest_file.read_text(encoding="utf-8").strip()
    metadata_path = config.model_dir / model_file.replace(".joblib", ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return AppState(
        config=config,
        scorer=scorer,
        model_version=metadata.get("version", config.version),
        model_file=model_file,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.service = _load_state()
        logger.info("Loaded model %s", app.state.service.model_file)
    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        app.state.service = None
    yield


app = FastAPI(
    title="East Africa Credit Scoring API",
        description=(
            "Credit scoring for M-Pesa, SACCO, bank, and mobile digital lenders "
            "(Tala, Branch, Zenka, etc.) with SHAP explainability and audit trails."
        ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_state() -> AppState:
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py before starting the API.",
        )
    return service


def _build_response(decision, shap_explanation, audit_id, model_version) -> ScoreResponse:
    shap_payload = None
    if shap_explanation is not None:
        shap_payload = ShapExplanationResponse(
            base_value=shap_explanation.base_value,
            predicted_log_odds=shap_explanation.predicted_log_odds,
            summary=shap_explanation.summary,
            contributions=[
                FeatureContributionResponse(
                    feature=item.feature,
                    raw_value=item.raw_value,
                    shap_value=item.shap_value,
                    impact=item.impact,
                )
                for item in shap_explanation.contributions
            ],
        )

    return ScoreResponse(
        applicant_id=decision.applicant_id,
        channel=decision.channel,
        probability_of_default=decision.probability_of_default,
        credit_score=decision.credit_score,
        decision=decision.decision.value,
        policy=PolicyResponse(
            passed=decision.policy.passed,
            reasons=list(decision.policy.reasons),
        ),
        top_risk_factors=decision.top_risk_factors,
        shap=shap_payload,
        audit_id=audit_id,
        model_version=model_version,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = getattr(app.state, "service", None)
    return HealthResponse(
        status="ok" if service else "degraded",
        model_loaded=service is not None,
        model_version=service.model_version if service else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    state = _get_state()
    metadata_path = state.config.model_dir / state.model_file.replace(".joblib", ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ModelInfoResponse(
        project=metadata["project"],
        version=metadata["version"],
        model_file=state.model_file,
        feature_count=len(metadata["feature_columns"]),
        decision_bands=metadata["decision_bands"],
        scorecard=metadata["scorecard"],
    )


@app.post("/score", response_model=ScoreResponse)
def score_applicant(request: ScoreRequest) -> ScoreResponse:
    state = _get_state()
    applicant = request.to_applicant()
    decision, shap_explanation, audit_id = state.scorer.score_with_audit(
        applicant,
        include_shap=request.include_shap,
        persist_audit_trail=request.persist_audit_trail,
        request_snapshot=request.snapshot(),
    )
    return _build_response(decision, shap_explanation, audit_id, state.model_version)


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(request: BatchScoreRequest) -> list[ScoreResponse]:
    state = _get_state()
    responses: list[ScoreResponse] = []
    for item in request.applicants:
        item = item.model_copy(
            update={
                "include_shap": request.include_shap,
                "persist_audit_trail": request.persist_audit_trail,
            }
        )
        applicant = item.to_applicant()
        decision, shap_explanation, audit_id = state.scorer.score_with_audit(
            applicant,
            include_shap=item.include_shap,
            persist_audit_trail=item.persist_audit_trail,
            request_snapshot=item.snapshot(),
        )
        responses.append(
            _build_response(decision, shap_explanation, audit_id, state.model_version)
        )
    return responses
