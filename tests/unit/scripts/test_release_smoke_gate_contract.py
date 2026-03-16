from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "smoke-release-gates.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "smoke-release-gates.ps1"


def test_release_smoke_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()


def test_bash_script_requires_private_origin_inputs_and_emits_artifact_names():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "APP_BASE_URL" in content
    assert "APP_PUBLIC_HOSTNAME" in content
    assert "API_BASE_URL" in content
    assert "API_PUBLIC_HOSTNAME" in content
    assert "PROBE_ORIGIN_URLS" in content
    assert "PHASE3_VALIDATION_ARTIFACT_NAME" in content
    assert "edge-origin-validation" in content
    assert "SMOKE_ARTIFACT_NAME" in content
    assert "SMOKE_PHASE_LABEL" in content
    assert "ROLLOUT_STATE_FILE" in content
    assert "validate-aca-phase3-auth.sh" in content
    assert "EXPECTED_CLIENT_ID" not in content
    assert "EXPECTED_ISSUER_URL" not in content
    assert "EXPECTED_ALLOWED_AUDIENCES" not in content
    assert "WRONG_AUDIENCE_TOKEN" not in content
    assert "PRODUCTION_REJECTION_TOKEN" not in content


def test_bash_script_covers_frontend_and_private_origin_endpoint_contracts():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "/api/health" in content
    assert "/health" in content
    assert "/health/readiness" in content
    assert "/api/collections" in content
    assert "/api/index-jobs" in content
    assert "/search/agent/stream" in content
    assert "/.auth/me" not in content


def test_powershell_script_requires_private_origin_inputs_and_reuses_validator():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "AppBaseUrl" in content
    assert "AppPublicHostname" in content
    assert "ApiBaseUrl" in content
    assert "ApiPublicHostname" in content
    assert "ProbeOriginUrls" in content
    assert "Phase3ValidationArtifactName" in content
    assert "edge-origin-validation" in content
    assert "SmokeArtifactName" in content
    assert "SmokePhaseLabel" in content
    assert "RolloutStateFile" in content
    assert "validate-aca-phase3-auth.ps1" in content
    assert "ExpectedClientId" not in content
    assert "ExpectedIssuerUrl" not in content
    assert "ExpectedAllowedAudiences" not in content
    assert "WrongAudienceToken" not in content
    assert "ProductionRejectionToken" not in content


def test_powershell_script_covers_frontend_and_private_origin_endpoint_contracts():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "/api/health" in content
    assert "/health" in content
    assert "/health/readiness" in content
    assert "/api/collections" in content
    assert "/api/index-jobs" in content
    assert "/search/agent/stream" in content
    assert "/.auth/me" not in content
