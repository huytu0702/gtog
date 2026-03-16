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
    assert 'az rest \\' in content
    assert '--body @"$tunnel_patch_file"' not in content
    assert '--command /bin/sh' not in content
    assert "validate_rollout_percentages" in content
    assert "CANARY_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "STABLE_TRAFFIC_PERCENT must be between 0 and 100" in content
    assert "CANARY_TRAFFIC_PERCENT and STABLE_TRAFFIC_PERCENT must sum to 100" in content
    assert 'echo ">>> Setting subscription: ${SUBSCRIPTION}"' in content
    assert "az account set --subscription \"$SUBSCRIPTION\"" in content
    assert content.index('if [[ "$ROLLOUT_MODE" == "promote" || "$ROLLOUT_MODE" == "rollback" ]]; then') < content.index('echo ">>> Registering providers"')



def test_bash_provision_script_supports_frontend_private_ingress_and_dual_host_contract():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "FRONTEND_APP_NAME" in content
    assert "FRONTEND_IMAGE" in content
    assert "APP_PUBLIC_HOSTNAME" in content
    assert "NEXT_PUBLIC_API_BASE_URL=https://${API_PUBLIC_HOSTNAME}" in content
    assert "CORS_ORIGINS=https://${APP_PUBLIC_HOSTNAME}" in content
    assert "REQUIRE_PLATFORM_AUTH=true" in content
    assert 'api_env_vars+=("EDGE_ORIGIN_SECRET=secretref:${EDGE_ORIGIN_SECRET_NAME}")' in content
    assert 'api_secret_args=()' in content
    assert 'local api_secret_args=()' not in content
    assert 'api_secret_args=(--secrets "${EDGE_ORIGIN_SECRET_NAME}=${EDGE_ORIGIN_SECRET}")' in content
    assert 'API_ARGS+=("${api_secret_args[@]}")' in content
    assert 'az containerapp secret set \\' in content
    assert 'az containerapp ingress cors update \\' in content
    assert '--allowed-origins "https://${APP_PUBLIC_HOSTNAME}"' in content
    assert "--allowed-methods GET HEAD OPTIONS POST PUT PATCH DELETE" in content
    assert "--allowed-headers '*'" in content
    assert "--allow-credentials true" in content
    assert "--max-age 600" in content
    assert "allowedExternalRedirectUrls" in content
    assert "authConfigs/current" in content
    assert "api-version=2025-07-01" in content
    assert 'graph_patch_body="$(<"$patch_file")"' in content
    assert '--body "$graph_patch_body"' in content
    assert '--body @"$patch_file"' not in content
    assert 'auth_config_patch_body="$(<"$auth_config_patch_file")"' in content
    assert '--body "$auth_config_patch_body"' in content
    assert '--body @"$auth_config_patch_file"' not in content
    assert '--token-store true' not in content
    assert '--tenant-id "$ENTRA_TENANT_ID"' not in content
    assert 'az containerapp auth microsoft update \\' not in content
    assert 'aad["login"]["loginParameters"] = json.loads(os.environ["EXPECTED_LOGIN_PARAMETERS_JSON"])' in content
    assert 'aad["registration"]["openIdIssuer"] = os.environ["ENTRA_ISSUER_URL"]' in content
    assert 'aad["validation"]["allowedAudiences"] = json.loads(os.environ["EXPECTED_ALLOWED_AUDIENCES_JSON"])' in content
    assert 'auth["identityProviders"] = {"azureActiveDirectory": aad}' in content
    assert "GOOGLE_CLIENT_ID" not in content
    assert "GOOGLE_CLIENT_SECRET" not in content
    assert "GOOGLE_CLIENT_SECRET_NAME" not in content
    assert "GOOGLE_ALLOWED_AUDIENCES" not in content
    assert "GOOGLE_LOGIN_SCOPES_JSON" not in content
    assert "EXPECTED_GOOGLE_ALLOWED_AUDIENCES_JSON" not in content
    assert "EXPECTED_GOOGLE_LOGIN_SCOPES_JSON" not in content
    assert "require_google_easy_auth_contract" not in content
    assert 'google["enabled"] = True' not in content
    assert 'identity_providers["google"] = google' not in content
    assert 'auth = auth.get("properties", auth)' in content
    assert 'az containerapp env update \\' in content
    assert '--public-network-access Disabled' in content
    assert '--excluded-paths "/health,/health/readiness"' in content
    assert 'continuing with existing environment settings' in content
    assert 'EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="[]"' in content
    assert 'EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="$(csv_to_json_array "https://${APP_PUBLIC_HOSTNAME}")"' in content
    assert "--target-port 3000" in content
    assert "Add public hostnames app.<domain> and api.<domain> to the tunnel." in content
    assert "Point app.<domain> to the frontend private origin in ACA." in content
    assert "Point api.<domain> to the API private origin in ACA." in content



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
    assert 'az rest `' in content
    assert '--command "/bin/sh"' not in content
    assert "Assert-RolloutPercentages" in content
    assert "CanaryTrafficPercent must be between 0 and 100." in content
    assert "StableTrafficPercent must be between 0 and 100." in content
    assert "CanaryTrafficPercent and StableTrafficPercent must sum to 100." in content
    assert 'Write-Host ">>> Setting subscription: $Subscription"' in content
    assert "az account set --subscription $Subscription --output none" in content
    assert content.index('if ($RolloutMode -in @("promote", "rollback"))') < content.index('Write-Host ">>> Registering providers"')



def test_powershell_provision_script_supports_frontend_private_ingress_and_dual_host_contract():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "FrontendAppName" in content
    assert "FrontendImage" in content
    assert "AppPublicHostname" in content
    assert "NEXT_PUBLIC_API_BASE_URL=https://$ApiPublicHostname" in content
    assert "CORS_ORIGINS=https://$AppPublicHostname" in content
    assert "REQUIRE_PLATFORM_AUTH=true" in content
    assert '$apiEnvVars += "EDGE_ORIGIN_SECRET=secretref:$EdgeOriginSecretName"' in content
    assert '$apiSecretArgs = @()' in content
    assert '$apiSecretArgs = @("--secrets", "$EdgeOriginSecretName=$EdgeOriginSecret")' in content
    assert '$apiArgs += $apiSecretArgs' in content
    assert "az containerapp secret set" in content
    assert "az containerapp ingress cors update" in content
    assert '--allowed-origins "https://$AppPublicHostname"' in content
    assert '--allowed-methods GET HEAD OPTIONS POST PUT PATCH DELETE' in content
    assert '--allowed-headers "*"' in content
    assert '--allow-credentials true' in content
    assert '--max-age 600' in content
    assert "allowedExternalRedirectUrls" in content
    assert "authConfigs/current" in content
    assert "api-version=2025-07-01" in content
    assert '--token-store true' not in content
    assert '"--tenant-id", $EntraTenantId' not in content
    assert 'auth", "microsoft", "update"' not in content
    assert '$aadLogin["loginParameters"] = @($ExpectedLoginParameters)' in content
    assert '$aadRegistration["openIdIssuer"] = $EntraIssuerUrl' in content
    assert '$aadValidation["allowedAudiences"] = @($ExpectedAllowedAudiences)' in content
    assert '$authProperties.identityProviders = @{ azureActiveDirectory = $azureActiveDirectory }' in content
    assert "GoogleClientId" not in content
    assert "GoogleClientSecret" not in content
    assert "GoogleClientSecretName" not in content
    assert "GoogleAllowedAudiences" not in content
    assert "GoogleLoginScopesJson" not in content
    assert "Assert-GoogleEasyAuthContract" not in content
    assert "ExpectedGoogleAllowedAudiences" not in content
    assert "ExpectedGoogleLoginScopes" not in content
    assert '$google["enabled"] = $true' not in content
    assert '$identityProviders["google"] = $google' not in content
    assert '$auth = if ($auth.properties) { $auth.properties } else { $auth }' in content
    assert 'az containerapp env update' in content
    assert '--public-network-access Disabled' in content
    assert '--excluded-paths "/health,/health/readiness"' in content
    assert 'continuing with existing environment settings' in content
    assert '$ExpectedAllowedExternalRedirectUrls = @()' in content
    assert '$ExpectedAllowedExternalRedirectUrls = @("https://$AppPublicHostname")' in content
    assert '--target-port", "3000"' in content or "--target-port 3000" in content
    assert "Add public hostnames app.<domain> and api.<domain> to the tunnel." in content
    assert "Point app.<domain> to the frontend private origin in ACA." in content
    assert "Point api.<domain> to the API private origin in ACA." in content



def test_bash_provision_script_requires_precise_public_hostname_guards_for_runtime_reconciliation():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "require_frontend_runtime_contract_hostnames" in content
    assert "require_api_runtime_contract_hostname" in content
    assert "require_easy_auth_hostname" in content
    assert "require_var APP_PUBLIC_HOSTNAME" in content
    assert "require_var API_PUBLIC_HOSTNAME" in content
    assert 'if container_app_exists "$FRONTEND_APP_NAME"; then' in content
    assert 'if container_app_exists "$API_APP_NAME"; then' in content
    assert 'if bool_true "$CONFIGURE_EASY_AUTH"; then' in content
    assert 'EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="$(csv_to_json_array "https://${APP_PUBLIC_HOSTNAME}")"' in content



def test_powershell_provision_script_requires_precise_public_hostname_guards_for_runtime_reconciliation():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "Assert-FrontendRuntimeContractHostnames" in content
    assert "Assert-ApiRuntimeContractHostname" in content
    assert "Assert-EasyAuthHostname" in content
    assert 'Require-Value -Name "AppPublicHostname" -Value $AppPublicHostname' in content
    assert 'Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname' in content
    assert 'if (Test-ContainerAppExists -Name $FrontendAppName) {' in content
    assert 'if (Test-ContainerAppExists -Name $ApiAppName) {' in content
    assert 'if ($ConfigureEasyAuth) {' in content
    assert '$ExpectedAllowedExternalRedirectUrls = @("https://$AppPublicHostname")' in content
