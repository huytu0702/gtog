from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "smoke-release-gates.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "smoke-release-gates.ps1"



def test_release_smoke_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()



def test_bash_script_requires_phase3_gate_inputs_and_emits_artifact_names():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "WRONG_AUDIENCE_TOKEN" in content
    assert "PRODUCTION_REJECTION_TOKEN" in content
    assert "PROBE_ORIGIN_URLS" in content
    assert "phase3-auth-origin-validation" in content
    assert "smoke-staging-report" in content
    assert "validate-aca-phase3-auth.sh" in content



def test_bash_script_covers_phase5_endpoint_contracts():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "/health" in content
    assert "/health/readiness" in content
    assert "/.auth/me" in content
    assert "/api/collections" in content
    assert "/api/index-jobs" in content
    assert "/search/agent/stream" in content



def test_powershell_script_requires_phase3_gate_inputs_and_reuses_validator():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "WrongAudienceToken" in content
    assert "ProductionRejectionToken" in content
    assert "ProbeOriginUrls" in content
    assert "phase3-auth-origin-validation" in content
    assert "smoke-staging-report" in content
    assert "validate-aca-phase3-auth.ps1" in content
