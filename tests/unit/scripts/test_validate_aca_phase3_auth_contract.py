from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "validate-aca-phase3-auth.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "validate-aca-phase3-auth.ps1"



def test_phase3_validator_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()



def test_bash_phase3_validator_enforces_auth_redirect_and_cors_contracts():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "EXPECTED_ALLOWED_AUDIENCES" in content
    assert "EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS" in content
    assert 'EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS_JSON="$(csv_to_json_array "$EXPECTED_ALLOWED_EXTERNAL_REDIRECT_URLS")"' in content
    assert 'AUTH_JSON="$(az rest --method get --uri "https://management.azure.com${AUTH_CONFIG_ID}?api-version=2025-07-01" --output json)"' in content
    assert 'auth.get("properties", auth)' in content
    assert 'expected_app_origin = expected_allowed_external_redirect_urls[0]' in content
    assert 'login.get("allowedExternalRedirectUrls") or []' in content
    assert 'if sorted(login.get("allowedExternalRedirectUrls") or []) != sorted(expected_allowed_external_redirect_urls):' in content
    assert 'errors.append("Easy Auth allowed external redirect URLs do not match the expected app origin")' in content
    assert 'if sorted(cors_policy.get("allowedOrigins") or []) != [expected_app_origin]:' in content
    assert 'errors.append("Ingress CORS allowed origins do not match the expected app origin")' in content
    assert 'if cors_policy.get("allowCredentials") is not True:' in content
    assert 'if sorted(cors_policy.get("allowedMethods") or []) != ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]:' in content
    assert 'if sorted(cors_policy.get("allowedHeaders") or []) != ["*"]:' in content
    assert 'if cors_policy.get("maxAge") != 600:' in content
    assert 'Expected unauthenticated /api/* request to return 401' in content
    assert 'Expected wrong-audience token to be rejected' in content
    assert 'Expected cross-environment token to be rejected' in content



def test_powershell_phase3_validator_enforces_auth_redirect_and_cors_contracts():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "ExpectedAllowedAudiences" in content
    assert "ExpectedAllowedExternalRedirectUrls" in content
    assert '$expectedAllowedExternalRedirectUrlArray = Convert-CsvToArray -Value $ExpectedAllowedExternalRedirectUrls' in content
    assert '$authConfigId = "$($apiApp.id)/authConfigs/current"' in content
    assert 'az rest --method get --uri "https://management.azure.com$authConfigId?api-version=2025-07-01" --output json' in content
    assert '$authProperties = if ($auth.properties) { $auth.properties } else { $auth }' in content
    assert '$actualAllowedExternalRedirectUrls = @($authProperties.login.allowedExternalRedirectUrls) | Sort-Object' in content
    assert '$corsAllowedOrigins = @($apiApp.properties.configuration.ingress.corsPolicy.allowedOrigins) | Sort-Object' in content
    assert '$errors.Add("Easy Auth allowed external redirect URLs do not match the expected app origin.")' in content
    assert '$errors.Add("Ingress CORS allowed origins do not match the expected app origin.")' in content
    assert '$errors.Add("Ingress CORS allowCredentials is not enabled.")' in content
    assert '$errors.Add("Ingress CORS allowed methods do not match the expected browser contract.")' in content
    assert '$errors.Add("Ingress CORS allowed headers do not match the expected browser contract.")' in content
    assert '$errors.Add("Ingress CORS maxAge is not 600.")' in content
    assert 'Expected unauthenticated /api/* request to return 401' in content
    assert 'Expected wrong-audience token to be rejected' in content
    assert 'Expected cross-environment token to be rejected' in content
