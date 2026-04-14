"""Study endpoints for broader real-data validation workflows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from web.config import settings
from web.dependencies import CurrentUser
from web.schemas import (
    ArtifactContentResponse,
    ValidationCampaignRequest,
    ValidationCampaignResponse,
)
from web.services.study_service import run_web_validation_campaign
from web.utils import assert_path_within, sanitize_filename

router = APIRouter(prefix="/api/studies", tags=["studies"])


def _resolve_campaign_artifact(campaign_id: str, artifact_name: str) -> Path:
    safe_id = sanitize_filename(campaign_id)
    safe_name = sanitize_filename(artifact_name)
    campaign_dir = settings.output_dir / "validation_campaigns" / safe_id
    artifact_path = campaign_dir / safe_name
    assert_path_within(artifact_path, campaign_dir)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact '{safe_name}' not found")
    if artifact_path.suffix.lower() not in {".json", ".md", ".csv"}:
        raise HTTPException(status_code=400, detail="Unsupported artifact type")
    return artifact_path


@router.post("/validation-campaign", response_model=ValidationCampaignResponse)
async def run_validation_study(
    body: ValidationCampaignRequest,
    _user: CurrentUser,
) -> ValidationCampaignResponse:
    """Run a broader multi-observation validation campaign."""
    try:
        payload = await asyncio.to_thread(
            run_web_validation_campaign,
            fits_filenames=body.fits_filenames,
            algorithm=body.algorithm,
            max_iterations=body.max_iterations,
            tolerance=body.tolerance,
            beta=body.beta,
            beta_schedule=body.beta_schedule,
            momentum=body.momentum,
            tv_weight=body.tv_weight,
            noise_model=body.noise_model,
            n_starts=body.n_starts,
            uncertainty_samples=body.uncertainty_samples,
            admm_rho=body.admm_rho,
            wf_step_size=body.wf_step_size,
            wf_spectral_init=body.wf_spectral_init,
            spectral_init=body.spectral_init,
            regulariser=body.regulariser,
            proximal_weight=body.proximal_weight,
            sparsity_threshold=body.sparsity_threshold,
            sparsity_keep_fraction=body.sparsity_keep_fraction,
            grid_size=body.grid_size,
        )
        return ValidationCampaignResponse.model_validate(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/validation-campaigns/{campaign_id}/artifacts/{artifact_name}",
    response_model=ArtifactContentResponse,
)
def get_validation_campaign_artifact(
    campaign_id: str,
    artifact_name: str,
    _user: CurrentUser,
) -> ArtifactContentResponse:
    """Return parsed validation-campaign artifact content."""
    artifact_path = _resolve_campaign_artifact(campaign_id, artifact_name)
    text = artifact_path.read_text(encoding="utf-8")
    if artifact_path.suffix.lower() == ".json":
        return ArtifactContentResponse(name=artifact_name, format="json", content=json.loads(text))
    if artifact_path.suffix.lower() == ".md":
        return ArtifactContentResponse(name=artifact_name, format="markdown", content=text)
    return ArtifactContentResponse(name=artifact_name, format="csv", content=text)
