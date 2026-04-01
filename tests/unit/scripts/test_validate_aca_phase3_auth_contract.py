from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "validate-aca-phase3-auth.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "validate-aca-phase3-auth.ps1"


def test_phase3_validator_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()


def test_bash_phase3_validator_enforces_private_origin_and_tunnel_contracts():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "APP_PUBLIC_HOSTNAME" in content
    assert "API_PUBLIC_HOSTNAME" in content
    assert "TUNNEL_SECRET_REF_NAME" in content
    assert "PROBE_ORIGIN_URLS" in content
    assert "ORIGIN_BYPASS_WORKSPACE" in content
    assert 'errors.append("Ingress CORS allowed origins do not match the expected app origin")' in content
    assert 'errors.append("API app REQUIRE_EDGE_AUTH must be true")' in content
    assert 'errors.append("Worker app still exposes ingress")' in content
    assert 'errors.append("Tunnel app should not expose ingress")' in content
    assert 'if args != ["tunnel", "--no-autoupdate", "run"]:' in content
    assert 'f"Tunnel app TUNNEL_TOKEN must reference the {expected_tunnel_secret_ref_name!r} secret"' in content
    assert "Direct-origin probe reached an HTTP handler" in content
    assert "Phase 3 private-origin validation checks completed successfully" in content
    assert "EXPECTED_CLIENT_ID" not in content
    assert "EXPECTED_ISSUER_URL" not in content
    assert "EXPECTED_ALLOWED_AUDIENCES" not in content
    assert "WRONG_AUDIENCE_TOKEN" not in content
    assert "PRODUCTION_REJECTION_TOKEN" not in content
    assert "/.auth/me" not in content
    assert "authConfigs/current" not in content
    assert "az containerapp auth microsoft show" not in content


def test_powershell_phase3_validator_enforces_private_origin_and_tunnel_contracts():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "AppPublicHostname" in content
    assert "ApiPublicHostname" in content
    assert "TunnelSecretRefName" in content
    assert "ProbeOriginUrls" in content
    assert "OriginBypassWorkspace" in content
    assert '$errors.Add("Ingress CORS allowed origins do not match the expected app origin.")' in content
    assert '$errors.Add("API app REQUIRE_EDGE_AUTH must be true.")' in content
    assert '$errors.Add("Worker app still exposes ingress.")' in content
    assert '$errors.Add("Tunnel app should not expose ingress.")' in content
    assert 'if ((@($tunnelContainer.args) -join "|") -ne "tunnel|--no-autoupdate|run")' in content
    assert '$errors.Add("Tunnel app TUNNEL_TOKEN must reference the $TunnelSecretRefName secret.")' in content
    assert "Direct-origin probe reached an HTTP handler" in content
    assert "Phase 3 private-origin validation checks completed successfully" in content
    assert "ExpectedClientId" not in content
    assert "ExpectedIssuerUrl" not in content
    assert "ExpectedAllowedAudiences" not in content
    assert "WrongAudienceToken" not in content
    assert "ProductionRejectionToken" not in content
    assert "/.auth/me" not in content
    assert "authConfigs/current" not in content
    assert "auth microsoft show" not in content
