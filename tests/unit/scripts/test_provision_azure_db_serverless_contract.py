from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASH_SCRIPT = REPO_ROOT / "scripts" / "provision-azure-db.sh"
POWERSHELL_SCRIPT = REPO_ROOT / "scripts" / "provision-azure-db.ps1"



def test_provision_db_scripts_exist():
    assert BASH_SCRIPT.exists()
    assert POWERSHELL_SCRIPT.exists()



def test_bash_db_provision_script_uses_serverless_cosmos_contract():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'LOCATION="${LOCATION:-southeastasia}"' in content
    assert "EnableServerless" in content
    assert "--capabilities EnableServerless" in content
    assert "already exists but is not configured for serverless" in content
    assert "capacity mode cannot be changed in place" in content
    assert "--max-throughput" not in content



def test_powershell_db_provision_script_uses_serverless_cosmos_contract():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert '[string]$Location = "southeastasia"' in content
    assert "EnableServerless" in content
    assert "--capabilities EnableServerless" in content
    assert "already exists but is not configured for serverless" in content
    assert "Capacity mode cannot be changed in place" in content
    assert "--max-throughput" not in content



def test_bash_db_provision_script_does_not_echo_secret_values():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "AZURE_STORAGE_CONNECTION_STRING=' not in content
    assert 'echo "AZURE_STORAGE_ACCOUNT_KEY=' not in content
    assert 'echo "AZURE_SEARCH_API_KEY=' not in content
    assert 'echo "AZURE_COSMOS_CONNECTION_STRING=' not in content
    assert 'echo "AZURE_COSMOS_KEY=' not in content
    assert "Retrieve secret values separately via Azure CLI before writing backend/.env." in content



def test_powershell_db_provision_script_does_not_echo_secret_values():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert 'Write-Host "AZURE_STORAGE_CONNECTION_STRING=' not in content
    assert 'Write-Host "AZURE_STORAGE_ACCOUNT_KEY=' not in content
    assert 'Write-Host "AZURE_SEARCH_API_KEY=' not in content
    assert 'Write-Host "AZURE_COSMOS_CONNECTION_STRING=' not in content
    assert 'Write-Host "AZURE_COSMOS_KEY=' not in content
    assert "Retrieve secret values separately via Azure CLI before writing backend/.env." in content



def test_bash_db_provision_script_avoids_retrieving_unused_search_and_cosmos_secrets():
    content = BASH_SCRIPT.read_text(encoding="utf-8")

    assert 'SEARCH_API_KEY="$(' not in content
    assert 'COSMOS_KEY="$(' not in content
    assert 'COSMOS_CONNECTION_STRING="$(' not in content



def test_powershell_db_provision_script_avoids_retrieving_unused_search_and_cosmos_secrets():
    content = POWERSHELL_SCRIPT.read_text(encoding="utf-8")

    assert '$searchApiKey = az search admin-key show' not in content
    assert '$cosmosKey = az cosmosdb keys list' not in content
    assert '$cosmosConnectionString = az cosmosdb keys list' not in content
