param(
    [string]$ResourceGroup = "rg-gtog-prod",
    [string]$Location = "southeastasia",
    [string]$Subscription = "1095803e-80bf-47e0-961f-3d74cb4c605c",
    [string]$ContainerAppEnvironment = "cae-gtog-prod",
    [string]$InfrastructureResourceGroup = "rg-gtog-prod-aca-infra",
    [string]$LogAnalyticsWorkspace = "law-gtog-prod",
    [string]$VnetName = "vnet-gtog-prod-aca",
    [string]$InfrastructureSubnetName = "snet-aca-infra",
    [string]$InfrastructureSubnetPrefix = "10.30.0.0/23",
    [string]$PrivateEndpointSubnetName = "snet-aca-private-endpoints",
    [string]$PrivateEndpointSubnetPrefix = "10.30.2.0/27",
    [string]$PrivateEndpointName = "pe-cae-gtog-prod",
    [string]$PrivateDnsZone = "",
    [string]$PrivateDnsLinkName = "link-cae-gtog-prod",
    [string]$ApiAppName = "ca-gtog-api-prod",
    [string]$WorkerAppName = "ca-gtog-worker-prod",
    [string]$TunnelAppName = "ca-gtog-tunnel-prod",
    [string]$ApiImage = "",
    [string]$WorkerImage = "",
    [string]$TunnelImage = "cloudflare/cloudflared:latest",
    [string]$UserAssignedIdentityName = "",
    [string]$KeyVaultName = "",
    [string]$TunnelToken = "",
    [string]$TunnelTokenSecretName = "cloudflare-tunnel-token",
    [string]$EdgeOriginSecret = "",
    [string]$EdgeOriginSecretName = "edge-origin-secret",
    [switch]$CreateApps
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = (Join-Path (Get-Location) ".azure")
}
New-Item -ItemType Directory -Path $env:AZURE_CONFIG_DIR -Force | Out-Null

if (-not $PrivateDnsZone) {
    $PrivateDnsZone = "privatelink.$Location.azurecontainerapps.io"
}

function Test-AzCommand {
    param([scriptblock]$Command)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    return ($exitCode -eq 0)
}

function Ensure-Subnet {
    param(
        [string]$Name,
        [string]$Prefix,
        [string]$Delegation
    )

    if (-not (Test-AzCommand {
        az network vnet subnet show `
            --resource-group $ResourceGroup `
            --vnet-name $VnetName `
            --name $Name `
            --output none 2>$null
    })) {
        $arguments = @(
            "network", "vnet", "subnet", "create",
            "--resource-group", $ResourceGroup,
            "--vnet-name", $VnetName,
            "--name", $Name,
            "--address-prefixes", $Prefix,
            "--output", "none"
        )
        if ($Delegation) {
            $arguments += @("--delegations", $Delegation)
        }
        az @arguments | Out-Null
    }
}

function Upsert-KeyVaultSecret {
    param(
        [string]$VaultName,
        [string]$SecretName,
        [string]$SecretValue
    )

    if (-not $VaultName -or -not $SecretValue) {
        return
    }

    az keyvault secret set `
        --vault-name $VaultName `
        --name $SecretName `
        --value $SecretValue `
        --output none
}

Write-Host ">>> Checking Azure login context..."
if (-not (Test-AzCommand { az account show --output none 2>$null })) {
    throw "Azure CLI is not logged in. Run: az login --use-device-code"
}

Write-Host ">>> Ensuring containerapp extension"
az extension add --name containerapp --upgrade --allow-preview true --output none

Write-Host ">>> Setting subscription: $Subscription"
az account set --subscription $Subscription --output none

Write-Host ">>> Registering providers"
az provider register --namespace Microsoft.App --wait --output none
az provider register --namespace Microsoft.OperationalInsights --wait --output none
az provider register --namespace Microsoft.Network --wait --output none

Write-Host ">>> Ensuring resource group: $ResourceGroup"
az group create --name $ResourceGroup --location $Location --output none

Write-Host ">>> Ensuring Log Analytics workspace: $LogAnalyticsWorkspace"
if (-not (Test-AzCommand {
    az monitor log-analytics workspace show `
        --resource-group $ResourceGroup `
        --workspace-name $LogAnalyticsWorkspace `
        --output none 2>$null
})) {
    az monitor log-analytics workspace create `
        --resource-group $ResourceGroup `
        --workspace-name $LogAnalyticsWorkspace `
        --location $Location `
        --output none
}

$workspaceId = az monitor log-analytics workspace show `
    --resource-group $ResourceGroup `
    --workspace-name $LogAnalyticsWorkspace `
    --query customerId `
    --output tsv
$workspaceKey = az monitor log-analytics workspace get-shared-keys `
    --resource-group $ResourceGroup `
    --workspace-name $LogAnalyticsWorkspace `
    --query primarySharedKey `
    --output tsv

Write-Host ">>> Ensuring VNet: $VnetName"
if (-not (Test-AzCommand {
    az network vnet show `
        --resource-group $ResourceGroup `
        --name $VnetName `
        --output none 2>$null
})) {
    az network vnet create `
        --resource-group $ResourceGroup `
        --name $VnetName `
        --location $Location `
        --address-prefixes "10.30.0.0/16" `
        --output none
}

Write-Host ">>> Ensuring infrastructure subnet"
Ensure-Subnet `
    -Name $InfrastructureSubnetName `
    -Prefix $InfrastructureSubnetPrefix `
    -Delegation "Microsoft.App/environments"

Write-Host ">>> Ensuring private endpoint subnet"
Ensure-Subnet `
    -Name $PrivateEndpointSubnetName `
    -Prefix $PrivateEndpointSubnetPrefix `
    -Delegation ""
az network vnet subnet update `
    --resource-group $ResourceGroup `
    --vnet-name $VnetName `
    --name $PrivateEndpointSubnetName `
    --disable-private-endpoint-network-policies true `
    --output none

$infraSubnetId = az network vnet subnet show `
    --resource-group $ResourceGroup `
    --vnet-name $VnetName `
    --name $InfrastructureSubnetName `
    --query id `
    --output tsv
$privateEndpointSubnetId = az network vnet subnet show `
    --resource-group $ResourceGroup `
    --vnet-name $VnetName `
    --name $PrivateEndpointSubnetName `
    --query id `
    --output tsv

Write-Host ">>> Ensuring ACA environment: $ContainerAppEnvironment"
if (-not (Test-AzCommand {
    az containerapp env show `
        --resource-group $ResourceGroup `
        --name $ContainerAppEnvironment `
        --output none 2>$null
})) {
    az containerapp env create `
        --resource-group $ResourceGroup `
        --name $ContainerAppEnvironment `
        --location $Location `
        --enable-workload-profiles true `
        --infrastructure-resource-group $InfrastructureResourceGroup `
        --infrastructure-subnet-resource-id $infraSubnetId `
        --internal-only true `
        --logs-workspace-id $workspaceId `
        --logs-workspace-key $workspaceKey `
        --output none
}

$environmentId = az containerapp env show `
    --resource-group $ResourceGroup `
    --name $ContainerAppEnvironment `
    --query id `
    --output tsv
$defaultDomain = az containerapp env show `
    --resource-group $ResourceGroup `
    --name $ContainerAppEnvironment `
    --query properties.defaultDomain `
    --output tsv

Write-Host ">>> Disabling ACA public network access"
$patchBody = "{""properties"":{""publicNetworkAccess"":""Disabled""}}"
az rest `
    --method patch `
    --uri "https://management.azure.com$environmentId?api-version=2024-03-01" `
    --body $patchBody `
    --headers "Content-Type=application/json" `
    --output none

Write-Host ">>> Ensuring private endpoint: $PrivateEndpointName"
if (-not (Test-AzCommand {
    az network private-endpoint show `
        --resource-group $ResourceGroup `
        --name $PrivateEndpointName `
        --output none 2>$null
})) {
    az network private-endpoint create `
        --resource-group $ResourceGroup `
        --name $PrivateEndpointName `
        --location $Location `
        --subnet $privateEndpointSubnetId `
        --private-connection-resource-id $environmentId `
        --group-id managedEnvironments `
        --connection-name "$PrivateEndpointName-connection" `
        --output none
}

Write-Host ">>> Ensuring private DNS zone: $PrivateDnsZone"
if (-not (Test-AzCommand {
    az network private-dns zone show `
        --resource-group $ResourceGroup `
        --name $PrivateDnsZone `
        --output none 2>$null
})) {
    az network private-dns zone create `
        --resource-group $ResourceGroup `
        --name $PrivateDnsZone `
        --output none
}

if (-not (Test-AzCommand {
    az network private-dns link vnet show `
        --resource-group $ResourceGroup `
        --zone-name $PrivateDnsZone `
        --name $PrivateDnsLinkName `
        --output none 2>$null
})) {
    az network private-dns link vnet create `
        --resource-group $ResourceGroup `
        --zone-name $PrivateDnsZone `
        --name $PrivateDnsLinkName `
        --virtual-network $VnetName `
        --registration-enabled false `
        --output none
}

$dnsGroupExists = Test-AzCommand {
    az network private-endpoint dns-zone-group show `
        --resource-group $ResourceGroup `
        --endpoint-name $PrivateEndpointName `
        --name "default" `
        --output none 2>$null
}
if (-not $dnsGroupExists) {
    az network private-endpoint dns-zone-group create `
        --resource-group $ResourceGroup `
        --endpoint-name $PrivateEndpointName `
        --name "default" `
        --private-dns-zone $PrivateDnsZone `
        --zone-name "default" `
        --output none
}

Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName $TunnelTokenSecretName -SecretValue $TunnelToken
Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName $EdgeOriginSecretName -SecretValue $EdgeOriginSecret

$identityResourceId = ""
if ($UserAssignedIdentityName) {
    $identityResourceId = az identity show `
        --resource-group $ResourceGroup `
        --name $UserAssignedIdentityName `
        --query id `
        --output tsv
}

if ($CreateApps) {
    if (-not $ApiImage) {
        throw "ApiImage is required when -CreateApps is used."
    }
    if (-not $WorkerImage) {
        throw "WorkerImage is required when -CreateApps is used."
    }
    if (-not $TunnelToken) {
        throw "TunnelToken is required when -CreateApps is used."
    }

    Write-Host ">>> Ensuring API app: $ApiAppName"
    if (-not (Test-AzCommand {
        az containerapp show `
            --resource-group $ResourceGroup `
            --name $ApiAppName `
            --output none 2>$null
    })) {
        $apiArgs = @(
            "containerapp", "create",
            "--resource-group", $ResourceGroup,
            "--name", $ApiAppName,
            "--environment", $ContainerAppEnvironment,
            "--image", $ApiImage,
            "--ingress", "internal",
            "--target-port", "8000",
            "--transport", "auto",
            "--cpu", "1.0",
            "--memory", "2.0Gi",
            "--min-replicas", "1",
            "--max-replicas", "2",
            "--output", "none"
        )
        if ($identityResourceId) {
            $apiArgs += @("--user-assigned", $identityResourceId)
        }
        az @apiArgs | Out-Null
    }

    Write-Host ">>> Ensuring worker app: $WorkerAppName"
    if (-not (Test-AzCommand {
        az containerapp show `
            --resource-group $ResourceGroup `
            --name $WorkerAppName `
            --output none 2>$null
    })) {
        $workerArgs = @(
            "containerapp", "create",
            "--resource-group", $ResourceGroup,
            "--name", $WorkerAppName,
            "--environment", $ContainerAppEnvironment,
            "--image", $WorkerImage,
            "--cpu", "1.0",
            "--memory", "2.0Gi",
            "--min-replicas", "1",
            "--max-replicas", "1",
            "--output", "none"
        )
        if ($identityResourceId) {
            $workerArgs += @("--user-assigned", $identityResourceId)
        }
        az @workerArgs | Out-Null
    }

    Write-Host ">>> Ensuring tunnel connector app: $TunnelAppName"
    if (-not (Test-AzCommand {
        az containerapp show `
            --resource-group $ResourceGroup `
            --name $TunnelAppName `
            --output none 2>$null
    })) {
        az containerapp create `
            --resource-group $ResourceGroup `
            --name $TunnelAppName `
            --environment $ContainerAppEnvironment `
            --image $TunnelImage `
            --cpu "0.5" `
            --memory "1.0Gi" `
            --min-replicas "2" `
            --max-replicas "2" `
            --secrets "tunnel-token=$TunnelToken" `
            --env-vars "TUNNEL_TOKEN=secretref:tunnel-token" `
            --command "/bin/sh" `
            --args "-c" "cloudflared tunnel --no-autoupdate run --token `"`$TUNNEL_TOKEN`"" `
            --output none
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Private-origin ACA provisioning complete."
Write-Host "=========================================="
Write-Host "ACA environment: $ContainerAppEnvironment"
Write-Host "Default private domain: $defaultDomain"
Write-Host "Private DNS zone: $PrivateDnsZone"
Write-Host ""
Write-Host "Next Cloudflare steps:"
Write-Host "1. Create a remotely managed tunnel for this environment."
Write-Host "2. Add public hostname api.<domain> to the tunnel."
Write-Host "3. Point the tunnel service to the API private origin in ACA."
Write-Host "4. If Easy Auth depends on the public host, set the origin request host header to api.<domain>."
Write-Host "5. Keep WAF, rate limiting, cache bypass, and optional X-Edge-Secret injection on api.<domain>."
