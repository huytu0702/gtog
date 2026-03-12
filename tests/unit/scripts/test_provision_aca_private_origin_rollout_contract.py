from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "provision-aca-private-origin.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "provision-aca-private-origin.ps1"


def test_provision_rollout_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()


def test_bash_provision_script_supports_recovery_and_rollout_contracts():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "API_MIN_REPLICAS" in content
    assert "WORKER_MIN_REPLICAS" in content
    assert "TUNNEL_MIN_REPLICAS" in content
    assert "ROLLOUT_MODE" in content
    assert "CANARY_TRAFFIC_PERCENT" in content
    assert "ROLLOUT_STATE_FILE" in content
    assert "az containerapp revision set-mode" in content
    assert "az containerapp ingress traffic set" in content
    assert 'cloudflared tunnel --no-autoupdate run --token "$TUNNEL_TOKEN"' in content
    assert "validate_rollout_percentages" in content
    assert "CANARY_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "STABLE_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "CANARY_TRAFFIC_PERCENT and STABLE_TRAFFIC_PERCENT must sum to 100" in content
    assert 'echo ">>> Setting subscription: ${SUBSCRIPTION}"' in content
    assert "az account set --subscription \"$SUBSCRIPTION\"" in content
    assert content.index('if [[ "$ROLLOUT_MODE" == "promote" || "$ROLLOUT_MODE" == "rollback" ]]; then') < content.index('echo ">>> Registering providers"')


def test_powershell_provision_script_supports_recovery_and_rollout_contracts():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "ApiMinReplicas" in content
    assert "WorkerMinReplicas" in content
    assert "TunnelMinReplicas" in content
    assert "RolloutMode" in content
    assert "CanaryTrafficPercent" in content
    assert "RolloutStateFile" in content
    assert "az containerapp revision set-mode" in content
    assert "az containerapp ingress traffic set" in content
    assert 'cloudflared tunnel --no-autoupdate run --token' in content
    assert "Assert-RolloutPercentages" in content
    assert "CanaryTrafficPercent must be between 0 and 100." in content
    assert "StableTrafficPercent must be between 0 and 100." in content
    assert "CanaryTrafficPercent and StableTrafficPercent must sum to 100." in content
    assert 'Write-Host ">>> Setting subscription: $Subscription"' in content
    assert "az account set --subscription $Subscription --output none" in content
    assert content.index('if ($RolloutMode -in @("promote", "rollback"))') < content.index('Write-Host ">>> Registering providers"')
