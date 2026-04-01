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
    [string]$FrontendAppName = "ca-gtog-frontend-prod",
    [string]$ApiAppName = "ca-gtog-api-prod",
    [string]$WorkerAppName = "ca-gtog-worker-prod",
    [string]$TunnelAppName = "ca-gtog-tunnel-prod",
    [string]$TunnelSecretRefName = "tunnel-token",
    [string]$FrontendImage = "",
    [string]$ApiImage = "",
    [string]$WorkerImage = "",
    [string]$TunnelImage = "cloudflare/cloudflared:latest",
    [string]$FrontendCpu = "1.0",
    [string]$FrontendMemory = "2.0Gi",
    [string]$FrontendMinReplicas = "1",
    [string]$FrontendMaxReplicas = "2",
    [string]$ApiCpu = "1.0",
    [string]$ApiMemory = "2.0Gi",
    [string]$ApiMinReplicas = "1",
    [string]$ApiMaxReplicas = "2",
    [string]$WorkerCpu = "1.0",
    [string]$WorkerMemory = "2.0Gi",
    [string]$WorkerMinReplicas = "1",
    [string]$WorkerMaxReplicas = "1",
    [string]$TunnelCpu = "0.5",
    [string]$TunnelMemory = "1.0Gi",
    [string]$TunnelMinReplicas = "2",
    [string]$TunnelMaxReplicas = "2",
    [ValidateSet("reconcile", "canary", "promote", "rollback")]
    [string]$RolloutMode = "reconcile",
    [int]$CanaryTrafficPercent = 10,
    [int]$StableTrafficPercent = 90,
    [string]$RolloutStateFile = "",
    [string]$UserAssignedIdentityName = "",
    [string]$KeyVaultName = "",
    [string]$TunnelToken = "",
    [string]$TunnelTokenSecretName = "cloudflare-tunnel-token",
    [string]$EdgeOriginSecret = "",
    [string]$EdgeOriginSecretName = "edge-origin-secret",
    [switch]$CreateApps,
    [string]$AppPublicHostname = "",
    [string]$ApiPublicHostname = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = (Join-Path (Get-Location) ".azure")
}
New-Item -ItemType Directory -Path $env:AZURE_CONFIG_DIR -Force | Out-Null

if (-not $PrivateDnsZone) {
    $PrivateDnsZone = "privatelink.$Location.azurecontainerapps.io"
}

$AccountTenantId = ""
$IdentityResourceId = ""
$IdentityPrincipalId = ""

function Test-AzCommand {
    param([scriptblock]$Command)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    return ($exitCode -eq 0)
}

function Require-Value {
    param(
        [string]$Name,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name is required for this operation."
    }
}

function Assert-FrontendRuntimeContractHostnames {
    Require-Value -Name "AppPublicHostname" -Value $AppPublicHostname
    Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname
}

function Assert-ApiRuntimeContractHostname {
    Require-Value -Name "AppPublicHostname" -Value $AppPublicHostname
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

function Test-ContainerAppExists {
    param([string]$Name)

    return (Test-AzCommand {
        az containerapp show `
            --resource-group $ResourceGroup `
            --name $Name `
            --output none 2>$null
    })
}

function Ensure-ContainerAppIdentity {
    param([string]$Name)

    if (-not $IdentityResourceId) {
        return
    }

    az containerapp identity assign `
        --resource-group $ResourceGroup `
        --name $Name `
        --user-assigned $IdentityResourceId `
        --output none | Out-Null
}

function Wait-ContainerAppProvisioning {
    param(
        [string]$Name,
        [int]$MaxAttempts = 40,
        [int]$DelaySeconds = 3
    )

    $lastState = ""
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $lastState = az containerapp show `
            --resource-group $ResourceGroup `
            --name $Name `
            --query properties.provisioningState `
            --output tsv 2>$null

        if ($LASTEXITCODE -eq 0) {
            if ($lastState -eq "Succeeded") {
                return
            }
            if ($lastState -eq "Failed") {
                throw "Container app $Name provisioning failed."
            }
        }

        Start-Sleep -Seconds $DelaySeconds
    }

    throw "Timed out waiting for container app $Name provisioning state to settle. Last state: $lastState"
}

function Ensure-FrontendIngressContract {
    $frontendArgs = @(
        "containerapp", "update",
        "--resource-group", $ResourceGroup,
        "--name", $FrontendAppName,
        "--cpu", $FrontendCpu,
        "--memory", $FrontendMemory,
        "--min-replicas", $FrontendMinReplicas,
        "--max-replicas", $FrontendMaxReplicas,
        "--set-env-vars", "NEXT_PUBLIC_API_BASE_URL=https://$ApiPublicHostname", "CORS_ORIGINS=https://$AppPublicHostname",
        "--output", "none"
    )
    if ($FrontendImage) {
        $frontendArgs += @("--image", $FrontendImage)
    }
    az @frontendArgs | Out-Null

    az containerapp ingress enable `
        --resource-group $ResourceGroup `
        --name $FrontendAppName `
        --type internal `
        --target-port 3000 `
        --transport auto `
        --output none | Out-Null
}

function Ensure-ApiIngressContract {
    $apiEnvVars = @(
        "APP_ROLE=api",
        "CORS_ORIGINS=https://$AppPublicHostname",
        "REQUIRE_EDGE_AUTH=true"
    )
    if ($EdgeOriginSecret) {
        az containerapp secret set `
            --resource-group $ResourceGroup `
            --name $ApiAppName `
            --secrets "$EdgeOriginSecretName=$EdgeOriginSecret" `
            --output none | Out-Null
        $apiEnvVars += "EDGE_ORIGIN_SECRET=secretref:$EdgeOriginSecretName"
    }

    $apiArgs = @(
        "containerapp", "update",
        "--resource-group", $ResourceGroup,
        "--name", $ApiAppName,
        "--cpu", $ApiCpu,
        "--memory", $ApiMemory,
        "--min-replicas", $ApiMinReplicas,
        "--max-replicas", $ApiMaxReplicas,
        "--set-env-vars"
    )
    $apiArgs += $apiEnvVars
    $apiArgs += @(
        "--output", "none"
    )
    if ($ApiImage) {
        $apiArgs += @("--image", $ApiImage)
    }
    az @apiArgs | Out-Null

    az containerapp ingress enable `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --type internal `
        --target-port 8000 `
        --transport auto `
        --output none | Out-Null

    az containerapp ingress cors update `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --allowed-origins "https://$AppPublicHostname" `
        --allowed-methods GET HEAD OPTIONS POST PUT PATCH DELETE `
        --allowed-headers "*" `
        --allow-credentials true `
        --max-age 600 `
        --output none | Out-Null
}

function Ensure-WorkerIngressContract {
    $workerArgs = @(
        "containerapp", "update",
        "--resource-group", $ResourceGroup,
        "--name", $WorkerAppName,
        "--cpu", $WorkerCpu,
        "--memory", $WorkerMemory,
        "--min-replicas", $WorkerMinReplicas,
        "--max-replicas", $WorkerMaxReplicas,
        "--set-env-vars", "APP_ROLE=worker",
        "--output", "none"
    )
    if ($WorkerImage) {
        $workerArgs += @("--image", $WorkerImage)
    }
    az @workerArgs | Out-Null

    az containerapp ingress disable `
        --resource-group $ResourceGroup `
        --name $WorkerAppName `
        --output none | Out-Null
}

function Ensure-TunnelConnectorContract {
    if (-not (Test-ContainerAppExists -Name $TunnelAppName)) {
        return
    }

    if ($TunnelToken) {
        az containerapp secret set `
            --resource-group $ResourceGroup `
            --name $TunnelAppName `
            --secrets "${TunnelSecretRefName}=$TunnelToken" `
            --output none | Out-Null
        Wait-ContainerAppProvisioning -Name $TunnelAppName
    }

    $tunnelPatch = @{
        properties = @{
            template = @{
                containers = @(
                    @{
                        name = $TunnelAppName
                        image = $TunnelImage
                        command = @()
                        args = @("tunnel", "--no-autoupdate", "run")
                        env = @(
                            @{
                                name = "TUNNEL_TOKEN"
                                secretRef = $TunnelSecretRefName
                            }
                        )
                        resources = @{
                            cpu = [double]$TunnelCpu
                            memory = $TunnelMemory
                        }
                    }
                )
                scale = @{
                    minReplicas = [int]$TunnelMinReplicas
                    maxReplicas = [int]$TunnelMaxReplicas
                }
            }
        }
    } | ConvertTo-Json -Depth 20 -Compress

    az rest `
        --method patch `
        --uri "https://management.azure.com/subscriptions/$Subscription/resourceGroups/$ResourceGroup/providers/Microsoft.App/containerApps/$TunnelAppName?api-version=2025-07-01" `
        --body $tunnelPatch `
        --headers "Content-Type=application/json" `
        --output none | Out-Null
    Wait-ContainerAppProvisioning -Name $TunnelAppName

    az containerapp ingress disable `
        --resource-group $ResourceGroup `
        --name $TunnelAppName `
        --output none | Out-Null
}

function Write-RolloutState {
    param(
        [string]$StateMode,
        [string]$StableRevision,
        [string]$CandidateRevision
    )

    if (-not $RolloutStateFile) {
        return
    }

    $rolloutStateDirectory = Split-Path -Path $RolloutStateFile -Parent
    if ($rolloutStateDirectory) {
        New-Item -ItemType Directory -Path $rolloutStateDirectory -Force | Out-Null
    }

    @{
        rollout_mode = $StateMode
        stable_revision = $StableRevision
        candidate_revision = $CandidateRevision
        canary_traffic_percent = $CanaryTrafficPercent
        stable_traffic_percent = $StableTrafficPercent
    } | ConvertTo-Json -Depth 10 | Set-Content -Path $RolloutStateFile
}

function Get-LatestRevisionName {
    return az containerapp revision list `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --query "sort_by([].{name:name, created:properties.createdTime}, &created)[-1].name" `
        --output tsv
}

function Get-StableRevisionName {
    param([string]$CandidateRevision)

    $stableRevision = az containerapp revision list `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --query "sort_by([?name!='${CandidateRevision}'].{name:name, created:properties.createdTime}, &created)[-1].name" `
        --output tsv
    if ($stableRevision) {
        return $stableRevision
    }

    return $CandidateRevision
}

function Read-RolloutStateField {
    param([string]$FieldName)

    if (-not $RolloutStateFile -or -not (Test-Path -Path $RolloutStateFile)) {
        throw "RolloutStateFile is required for $RolloutMode rollout mode."
    }

    $payload = Get-Content -Path $RolloutStateFile -Raw | ConvertFrom-Json
    return [string]($payload.$FieldName)
}

function Assert-RolloutPercentages {
    if ($CanaryTrafficPercent -lt 0 -or $CanaryTrafficPercent -gt 100) {
        throw "CanaryTrafficPercent must be between 0 and 100."
    }

    if ($StableTrafficPercent -lt 0 -or $StableTrafficPercent -gt 100) {
        throw "StableTrafficPercent must be between 0 and 100."
    }

    if (($CanaryTrafficPercent + $StableTrafficPercent) -ne 100) {
        throw "CanaryTrafficPercent and StableTrafficPercent must sum to 100."
    }
}

function Set-CanaryTrafficSplit {
    param(
        [string]$StableRevision,
        [string]$CandidateRevision
    )

    Assert-RolloutPercentages

    if ($StableRevision -eq $CandidateRevision) {
        throw "No previous stable revision found for canary traffic split."
    }

    az containerapp revision set-mode `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --mode multiple `
        --output none | Out-Null
    az containerapp revision activate `
        --resource-group $ResourceGroup `
        --revision $StableRevision `
        --output none | Out-Null
    az containerapp revision activate `
        --resource-group $ResourceGroup `
        --revision $CandidateRevision `
        --output none | Out-Null
    az containerapp ingress traffic set `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --revision-weight "${StableRevision}=$StableTrafficPercent" "${CandidateRevision}=$CanaryTrafficPercent" `
        --output none | Out-Null

    Write-RolloutState -StateMode "canary" -StableRevision $StableRevision -CandidateRevision $CandidateRevision
}

function Promote-FullTraffic {
    param(
        [string]$StableRevision,
        [string]$CandidateRevision
    )

    az containerapp revision set-mode `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --mode multiple `
        --output none | Out-Null
    az containerapp revision activate `
        --resource-group $ResourceGroup `
        --revision $CandidateRevision `
        --output none | Out-Null
    az containerapp ingress traffic set `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --revision-weight "${CandidateRevision}=100" `
        --output none | Out-Null

    Write-RolloutState -StateMode "promote" -StableRevision $StableRevision -CandidateRevision $CandidateRevision
}

function Rollback-ToStableTraffic {
    param(
        [string]$StableRevision,
        [string]$CandidateRevision
    )

    az containerapp revision set-mode `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --mode multiple `
        --output none | Out-Null
    az containerapp revision activate `
        --resource-group $ResourceGroup `
        --revision $StableRevision `
        --output none | Out-Null
    az containerapp ingress traffic set `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --revision-weight "${StableRevision}=100" "${CandidateRevision}=0" `
        --output none | Out-Null

    Write-RolloutState -StateMode "rollback" -StableRevision $StableRevision -CandidateRevision $CandidateRevision
}

Write-Host ">>> Checking Azure login context..."
if (-not (Test-AzCommand { az account show --output none 2>$null })) {
    throw "Azure CLI is not logged in. Run: az login --use-device-code"
}

Write-Host ">>> Ensuring containerapp extension"
az extension add --name containerapp --upgrade --allow-preview true --output none

Write-Host ">>> Setting subscription: $Subscription"
az account set --subscription $Subscription --output none
$AccountTenantId = az account show --query tenantId --output tsv

if ($RolloutMode -in @("promote", "rollback")) {
    $StableRevision = Read-RolloutStateField -FieldName "stable_revision"
    $CandidateRevision = Read-RolloutStateField -FieldName "candidate_revision"
} else {
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
    if (-not (Test-AzCommand {
        az containerapp env update `
            --resource-group $ResourceGroup `
            --name $ContainerAppEnvironment `
            --public-network-access Disabled `
            --output none
    })) {
        Write-Warning "Failed to disable ACA public network access via az containerapp env update; continuing with existing environment settings"
    }

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

    if ($UserAssignedIdentityName) {
        $IdentityResourceId = az identity show `
            --resource-group $ResourceGroup `
            --name $UserAssignedIdentityName `
            --query id `
            --output tsv
        $IdentityPrincipalId = az identity show `
            --resource-group $ResourceGroup `
            --name $UserAssignedIdentityName `
            --query principalId `
            --output tsv
    }

    if ($CreateApps) {
        if (-not $FrontendImage) {
            throw "FrontendImage is required when -CreateApps is used."
        }
        if (-not $ApiImage) {
            throw "ApiImage is required when -CreateApps is used."
        }
        if (-not $WorkerImage) {
            throw "WorkerImage is required when -CreateApps is used."
        }
        if (-not $AppPublicHostname) {
            throw "AppPublicHostname is required when -CreateApps is used."
        }
        if (-not $ApiPublicHostname) {
            throw "ApiPublicHostname is required when -CreateApps is used."
        }
        if (-not $TunnelToken) {
            throw "TunnelToken is required when -CreateApps is used."
        }

        Write-Host ">>> Ensuring frontend app: $FrontendAppName"
        if (-not (Test-ContainerAppExists -Name $FrontendAppName)) {
            $frontendArgs = @(
                "containerapp", "create",
                "--resource-group", $ResourceGroup,
                "--name", $FrontendAppName,
                "--environment", $ContainerAppEnvironment,
                "--image", $FrontendImage,
                "--ingress", "internal",
                "--target-port", "3000",
                "--transport", "auto",
                "--cpu", $FrontendCpu,
                "--memory", $FrontendMemory,
                "--min-replicas", $FrontendMinReplicas,
                "--max-replicas", $FrontendMaxReplicas,
                "--env-vars", "NEXT_PUBLIC_API_BASE_URL=https://$ApiPublicHostname", "CORS_ORIGINS=https://$AppPublicHostname",
                "--output", "none"
            )
            if ($IdentityResourceId) {
                $frontendArgs += @("--user-assigned", $IdentityResourceId)
            }
            az @frontendArgs | Out-Null
        }

        Write-Host ">>> Ensuring API app: $ApiAppName"
        if (-not (Test-ContainerAppExists -Name $ApiAppName)) {
                $apiEnvVars = @(
                    "APP_ROLE=api",
                    "CORS_ORIGINS=https://$AppPublicHostname",
                    "REQUIRE_EDGE_AUTH=true"
                )
            $apiSecretArgs = @()
            if ($EdgeOriginSecret) {
                $apiEnvVars += "EDGE_ORIGIN_SECRET=secretref:$EdgeOriginSecretName"
                $apiSecretArgs = @("--secrets", "$EdgeOriginSecretName=$EdgeOriginSecret")
            }

            $apiArgs = @(
                "containerapp", "create",
                "--resource-group", $ResourceGroup,
                "--name", $ApiAppName,
                "--environment", $ContainerAppEnvironment,
                "--image", $ApiImage,
                "--ingress", "internal",
                "--target-port", "8000",
                "--transport", "auto",
                "--cpu", $ApiCpu,
                "--memory", $ApiMemory,
                "--min-replicas", $ApiMinReplicas,
                "--max-replicas", $ApiMaxReplicas,
                "--env-vars"
            )
            $apiArgs += $apiEnvVars
            $apiArgs += $apiSecretArgs
            $apiArgs += @(
                "--output", "none"
            )
            if ($IdentityResourceId) {
                $apiArgs += @("--user-assigned", $IdentityResourceId)
            }
            az @apiArgs | Out-Null
        }

        Write-Host ">>> Ensuring worker app: $WorkerAppName"
        if (-not (Test-ContainerAppExists -Name $WorkerAppName)) {
            $workerArgs = @(
                "containerapp", "create",
                "--resource-group", $ResourceGroup,
                "--name", $WorkerAppName,
                "--environment", $ContainerAppEnvironment,
                "--image", $WorkerImage,
                "--cpu", $WorkerCpu,
                "--memory", $WorkerMemory,
                "--min-replicas", $WorkerMinReplicas,
                "--max-replicas", $WorkerMaxReplicas,
                "--env-vars", "APP_ROLE=worker",
                "--output", "none"
            )
            if ($IdentityResourceId) {
                $workerArgs += @("--user-assigned", $IdentityResourceId)
            }
            az @workerArgs | Out-Null
        }

        Write-Host ">>> Ensuring tunnel connector app: $TunnelAppName"
        if (-not (Test-ContainerAppExists -Name $TunnelAppName)) {
            az containerapp create `
                --resource-group $ResourceGroup `
                --name $TunnelAppName `
                --environment $ContainerAppEnvironment `
                --image $TunnelImage `
                --cpu $TunnelCpu `
                --memory $TunnelMemory `
                --min-replicas $TunnelMinReplicas `
                --max-replicas $TunnelMaxReplicas `
                --secrets "${TunnelSecretRefName}=$TunnelToken" `
                --env-vars "TUNNEL_TOKEN=secretref:$TunnelSecretRefName" `
                --output none
        }
    }

    if (Test-ContainerAppExists -Name $FrontendAppName) {
        Assert-FrontendRuntimeContractHostnames
        Write-Host ">>> Reconciling frontend ingress and runtime contract"
        Ensure-ContainerAppIdentity -Name $FrontendAppName
        Ensure-FrontendIngressContract
    }

    if (Test-ContainerAppExists -Name $ApiAppName) {
        Assert-ApiRuntimeContractHostname
        Write-Host ">>> Reconciling API ingress and runtime role"
        Ensure-ContainerAppIdentity -Name $ApiAppName
        Ensure-ApiIngressContract
    }

    if (Test-ContainerAppExists -Name $WorkerAppName) {
        Write-Host ">>> Reconciling worker ingress and runtime role"
        Ensure-ContainerAppIdentity -Name $WorkerAppName
        Ensure-WorkerIngressContract
    }

    if (Test-ContainerAppExists -Name $TunnelAppName) {
        Write-Host ">>> Reconciling tunnel connector contract"
        Ensure-TunnelConnectorContract
    }

    $CandidateRevision = Get-LatestRevisionName
    $StableRevision = Get-StableRevisionName -CandidateRevision $CandidateRevision
}

switch ($RolloutMode) {
    "canary" {
        Set-CanaryTrafficSplit -StableRevision $StableRevision -CandidateRevision $CandidateRevision
    }
    "promote" {
        Promote-FullTraffic -StableRevision $StableRevision -CandidateRevision $CandidateRevision
    }
    "rollback" {
        Rollback-ToStableTraffic -StableRevision $StableRevision -CandidateRevision $CandidateRevision
    }
    "reconcile" {
        Write-RolloutState -StateMode "reconcile" -StableRevision $StableRevision -CandidateRevision $CandidateRevision
    }
    default {
        throw "Unsupported RolloutMode: $RolloutMode"
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
Write-Host "2. Add public hostnames app.<domain> and api.<domain> to the tunnel."
Write-Host "3. Point app.<domain> to the frontend private origin in ACA."
Write-Host "4. Point api.<domain> to the API private origin in ACA."
Write-Host "5. Keep WAF, rate limiting, cache bypass, and optional X-Edge-Secret injection on api.<domain>."
Write-Host "6. Run scripts/validate-aca-phase3-auth.ps1 and docs/runbooks/origin-bypass-verification.md before promotion."
Write-Host "7. Confirm direct-origin probes fail at the network layer and tunnel failover remains healthy."
