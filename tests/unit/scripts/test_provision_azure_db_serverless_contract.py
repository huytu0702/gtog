from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "provision-azure-db.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "provision-azure-db.ps1"


def test_provision_db_scripts_exist() -> None:
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()


def test_bash_db_provision_script_uses_serverless_cosmos_contract() -> None:
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'LOCATION="${LOCATION:-southeastasia}"' in content
    assert "EnableServerless" in content
    assert "--capabilities EnableServerless" in content
    assert "exists but is NOT serverless" in content
    assert "Capacity mode cannot be changed in place" in content
    assert "--max-throughput" not in content


def test_powershell_db_provision_script_uses_serverless_cosmos_contract() -> None:
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert '[string]$Location         = "southeastasia"' in content
    assert "EnableServerless" in content
    assert "--capabilities" in content
    assert "EnableServerless" in content
    assert "exists but is NOT serverless" in content
    assert "Capacity mode cannot be changed in place" in content
    assert "--max-throughput" not in content


def test_bash_db_provision_script_does_not_echo_secret_values() -> None:
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "AZURE_STORAGE_CONNECTION_STRING=' not in content
    assert 'echo "AZURE_STORAGE_ACCOUNT_KEY=' not in content
    assert 'echo "AZURE_SEARCH_API_KEY=' not in content
    assert 'echo "AZURE_COSMOS_CONNECTION_STRING=' not in content
    assert 'echo "AZURE_COSMOS_KEY=' not in content
    assert "Retrieve secret values separately" in content


def test_powershell_db_provision_script_does_not_echo_secret_values() -> None:
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert 'Write-Host "AZURE_STORAGE_CONNECTION_STRING=' not in content
    assert 'Write-Host "AZURE_STORAGE_ACCOUNT_KEY=' not in content
    assert 'Write-Host "AZURE_SEARCH_API_KEY=' not in content
    assert 'Write-Host "AZURE_COSMOS_CONNECTION_STRING=' not in content
    assert 'Write-Host "AZURE_COSMOS_KEY=' not in content
    assert "Retrieve secret values separately" in content


def test_bash_db_provision_script_avoids_retrieving_unused_search_and_cosmos_secrets() -> None:
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'SEARCH_API_KEY="$(' not in content
    assert 'COSMOS_KEY="$(' not in content
    assert 'COSMOS_CONNECTION_STRING="$(' not in content


def test_powershell_db_provision_script_avoids_retrieving_unused_search_and_cosmos_secrets() -> None:
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "$searchApiKey = az search admin-key show" not in content
    assert "$cosmosKey = az cosmosdb keys list" not in content
    assert "$cosmosConnectionString = az cosmosdb keys list" not in content


def test_bash_db_provision_script_does_not_define_or_create_serving_containers() -> None:
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert "ENTITIES_CONTAINER" not in content
    assert "RELATIONSHIPS_CONTAINER" not in content
    assert "TEXT_UNITS_CONTAINER" not in content
    assert "COMMUNITIES_CONTAINER" not in content
    assert "COMMUNITY_REPORTS_CONTAINER" not in content
    assert "COVARIATES_CONTAINER" not in content
    assert ">>> Ensuring serving containers" not in content
    assert "AZURE_COSMOS_ENTITIES_CONTAINER" not in content
    assert "AZURE_COSMOS_RELATIONSHIPS_CONTAINER" not in content
    assert "AZURE_COSMOS_TEXT_UNITS_CONTAINER" not in content
    assert "AZURE_COSMOS_COMMUNITIES_CONTAINER" not in content
    assert "AZURE_COSMOS_COMMUNITY_REPORTS_CONTAINER" not in content
    assert "AZURE_COSMOS_COVARIATES_CONTAINER" not in content


def test_powershell_db_provision_script_does_not_define_or_create_serving_containers() -> None:
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert "$EntitiesContainer" not in content
    assert "$RelationshipsContainer" not in content
    assert "$TextUnitsContainer" not in content
    assert "$CommunitiesContainer" not in content
    assert "$CommunityReportsContainer" not in content
    assert "$CovariatesContainer" not in content
    assert ">>> Ensuring serving containers" not in content
    assert "AZURE_COSMOS_ENTITIES_CONTAINER" not in content
    assert "AZURE_COSMOS_RELATIONSHIPS_CONTAINER" not in content
    assert "AZURE_COSMOS_TEXT_UNITS_CONTAINER" not in content
    assert "AZURE_COSMOS_COMMUNITIES_CONTAINER" not in content
    assert "AZURE_COSMOS_COMMUNITY_REPORTS_CONTAINER" not in content
    assert "AZURE_COSMOS_COVARIATES_CONTAINER" not in content
