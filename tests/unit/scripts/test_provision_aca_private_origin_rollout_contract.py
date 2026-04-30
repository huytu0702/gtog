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
    assert '"command": []' in content
    assert '"args": ["tunnel", "--no-autoupdate", "run"]' in content
    assert 'tunnel_patch_body="$(<"$tunnel_patch_file")"' in content
    assert '--body "$tunnel_patch_body"' in content
    assert 'wait_for_container_app_provisioning "$TUNNEL_APP_NAME"' in content
    assert "validate_rollout_percentages" in content
    assert "CANARY_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "STABLE_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "CANARY_TRAFFIC_PERCENT and STABLE_TRAFFIC_PERCENT must sum to 100" in content


def test_bash_provision_script_supports_frontend_private_ingress_and_edge_auth_contract():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "FRONTEND_APP_NAME" in content
    assert "FRONTEND_IMAGE" in content
    assert "APP_PUBLIC_HOSTNAME" in content
    assert "NEXT_PUBLIC_API_BASE_URL=https://${API_PUBLIC_HOSTNAME}" in content
    assert "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}" in content
    assert "REQUIRE_EDGE_AUTH=true" in content
    assert 'api_env_vars+=("EDGE_ORIGIN_SECRET=secretref:${EDGE_ORIGIN_SECRET_NAME}")' in content
    assert 'api_secret_args=(--secrets "${EDGE_ORIGIN_SECRET_NAME}=${EDGE_ORIGIN_SECRET}")' in content
    assert 'API_ARGS+=("${api_secret_args[@]}")' in content
    assert 'az containerapp secret set \\' in content
    assert 'az containerapp ingress cors update \\' in content
    assert '--allowed-origins "https://${APP_PUBLIC_HOSTNAME}"' in content
    assert "--allowed-methods GET HEAD OPTIONS POST PUT PATCH DELETE" in content
    assert "--allowed-headers '*'" in content
    assert "--allow-credentials true" in content
    assert "--max-age 600" in content
    assert "az containerapp env update \\" in content
    assert "--public-network-access Disabled" in content
    assert "Add public hostnames app.<domain> and api.<domain> to the tunnel." in content
    assert "Point app.<domain> to the frontend private origin in ACA." in content
    assert "Point api.<domain> to the API private origin in ACA." in content
    assert "Run scripts/validate-aca-phase3-auth.sh" in content
    assert "CONFIGURE_EASY_AUTH" not in content
    assert "ensure_entra_app_contract" not in content
    assert "configure_easy_auth" not in content
    assert "verify_phase3_contract" not in content
    assert "authConfigs/current" not in content
    assert "allowedExternalRedirectUrls" not in content
    assert "az containerapp auth update" not in content
    assert "/.auth" not in content
    assert "REQUIRE_PLATFORM_AUTH=true" not in content


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
    assert 'command = @()' in content
    assert 'args = @("tunnel", "--no-autoupdate", "run")' in content
    assert 'Wait-ContainerAppProvisioning -Name $TunnelAppName' in content
    assert "Assert-RolloutPercentages" in content
    assert "CanaryTrafficPercent must be between 0 and 100." in content
    assert "StableTrafficPercent must be between 0 and 100." in content
    assert "CanaryTrafficPercent and StableTrafficPercent must sum to 100." in content


def test_powershell_provision_script_supports_frontend_private_ingress_and_edge_auth_contract():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "FrontendAppName" in content
    assert "FrontendImage" in content
    assert "AppPublicHostname" in content
    assert "NEXT_PUBLIC_API_BASE_URL=https://$ApiPublicHostname" in content
    assert "CORS_ORIGINS=https://$AppPublicHostname" in content
    assert "REQUIRE_EDGE_AUTH=true" in content
    assert '$apiEnvVars += "EDGE_ORIGIN_SECRET=secretref:$EdgeOriginSecretName"' in content
    assert '$apiSecretArgs = @("--secrets", "$EdgeOriginSecretName=$EdgeOriginSecret")' in content
    assert '$apiArgs += $apiSecretArgs' in content
    assert "az containerapp secret set" in content
    assert "az containerapp ingress cors update" in content
    assert '--allowed-origins "https://$AppPublicHostname"' in content
    assert '--allowed-methods GET HEAD OPTIONS POST PUT PATCH DELETE' in content
    assert '--allowed-headers "*"' in content
    assert '--allow-credentials true' in content
    assert '--max-age 600' in content
    assert "az containerapp env update" in content
    assert "--public-network-access Disabled" in content
    assert "Add public hostnames app.<domain> and api.<domain> to the tunnel." in content
    assert "Point app.<domain> to the frontend private origin in ACA." in content
    assert "Point api.<domain> to the API private origin in ACA." in content
    assert "Run scripts/validate-aca-phase3-auth.ps1" in content
    assert "ConfigureEasyAuth" not in content
    assert "Ensure-EntraAppContract" not in content
    assert "Configure-EasyAuth" not in content
    assert "Verify-Phase3Contract" not in content
    assert "authConfigs/current" not in content
    assert "allowedExternalRedirectUrls" not in content
    assert "auth update" not in content
    assert "/.auth" not in content
    assert "REQUIRE_PLATFORM_AUTH=true" not in content


def test_bash_provision_script_requires_precise_public_hostname_guards_for_runtime_reconciliation():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "require_frontend_runtime_contract_hostnames" in content
    assert "require_api_runtime_contract_hostname" in content
    assert "require_var APP_PUBLIC_HOSTNAME" in content
    assert "require_var API_PUBLIC_HOSTNAME" in content
    assert 'if container_app_exists "$FRONTEND_APP_NAME"; then' in content
    assert 'if container_app_exists "$API_APP_NAME"; then' in content
    assert "require_easy_auth_hostname" not in content
    assert 'if bool_true "$CONFIGURE_EASY_AUTH"; then' not in content


def test_powershell_provision_script_requires_precise_public_hostname_guards_for_runtime_reconciliation():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "Assert-FrontendRuntimeContractHostnames" in content
    assert "Assert-ApiRuntimeContractHostname" in content
    assert 'Require-Value -Name "AppPublicHostname" -Value $AppPublicHostname' in content
    assert 'Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname' in content
    assert 'if (Test-ContainerAppExists -Name $FrontendAppName) {' in content
    assert 'if (Test-ContainerAppExists -Name $ApiAppName) {' in content
    assert "Assert-EasyAuthHostname" not in content
    assert 'if ($ConfigureEasyAuth) {' not in content
