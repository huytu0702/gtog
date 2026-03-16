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
    [switch]$ConfigureEasyAuth,
    [switch]$CreateEntraApp,
    [switch]$ResetEntraClientSecret,
    [string]$AppPublicHostname = "",
    [string]$ApiPublicHostname = "",
    [string]$EntraAppDisplayName = "",
    [string]$EntraAppId = "",
    [string]$EntraTenantId = "",
    [string]$EntraIssuerUrl = "",
    [string]$EntraClientSecret = "",
    [string]$EntraClientSecretName = "entra-client-secret",
    [string]$EntraClientSecretDisplayName = "",
    [string]$ApiAppIdUri = "",
    [string]$AllowedAudiences = "",
    [string]$EntraScopeName = "access_as_user",
    [string]$EntraScopeAdminConsentDisplayName = "",
    [string]$EntraScopeAdminConsentDescription = "",
    [string]$EntraScopeUserConsentDisplayName = "",
    [string]$EntraScopeUserConsentDescription = "",
    [string]$AadLoginParametersJson = ""
)

$ErrorActionPreference = "Stop"

if (-not $env:AZURE_CONFIG_DIR) {
    $env:AZURE_CONFIG_DIR = (Join-Path (Get-Location) ".azure")
}
New-Item -ItemType Directory -Path $env:AZURE_CONFIG_DIR -Force | Out-Null

if (-not $PrivateDnsZone) {
    $PrivateDnsZone = "privatelink.$Location.azurecontainerapps.io"
}
if (-not $EntraClientSecretDisplayName) {
    $EntraClientSecretDisplayName = "$ApiAppName-easy-auth"
}
if (-not $EntraScopeAdminConsentDisplayName) {
    $EntraScopeAdminConsentDisplayName = "Access $ApiAppName"
}
if (-not $EntraScopeAdminConsentDescription) {
    $EntraScopeAdminConsentDescription = "Allow the signed-in user to access $ApiAppName."
}
if (-not $EntraScopeUserConsentDisplayName) {
    $EntraScopeUserConsentDisplayName = "Access $ApiAppName"
}
if (-not $EntraScopeUserConsentDescription) {
    $EntraScopeUserConsentDescription = "Allow the app to access $ApiAppName on your behalf."
}

$AccountTenantId = ""
$IdentityResourceId = ""
$IdentityPrincipalId = ""
$ExpectedAllowedAudiences = @()
$ExpectedAllowedExternalRedirectUrls = @()
$ExpectedLoginParameters = @()
$ExpectedScopeValue = ""
$ExpectedCallbackUrl = ""

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

function Assert-EasyAuthHostname {
    Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname
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

function Convert-CsvToArray {
    param([string]$Value)

    if (-not $Value) {
        return @()
    }

    return $Value.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

function Get-DefaultLoginParameters {
    param([string]$ScopeValue)

    return @("scope=openid profile email offline_access $ScopeValue")
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
        "REQUIRE_PLATFORM_AUTH=true"
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

function Ensure-EntraAppContract {
    Require-Value -Name "ApiPublicHostname" -Value $ApiPublicHostname

    if (-not $EntraAppId -and $EntraAppDisplayName) {
        $EntraAppId = az ad app list `
            --display-name $EntraAppDisplayName `
            --query "[?displayName=='$EntraAppDisplayName'] | [0].appId" `
            --output tsv
    }

    if (-not $EntraAppId) {
        if (-not $CreateEntraApp) {
            throw "EntraAppId or EntraAppDisplayName is required unless -CreateEntraApp is used."
        }
        Require-Value -Name "EntraAppDisplayName" -Value $EntraAppDisplayName
        Write-Host ">>> Creating Entra app registration: $EntraAppDisplayName"
        $EntraAppId = az ad app create `
            --display-name $EntraAppDisplayName `
            --sign-in-audience AzureADMyOrg `
            --query appId `
            --output tsv
    }

    $entraObjectId = az ad app show --id $EntraAppId --query id --output tsv

    if (-not $ApiAppIdUri) {
        $ApiAppIdUri = "api://$EntraAppId"
    }
    if (-not $AllowedAudiences) {
        $AllowedAudiences = $ApiAppIdUri
    }

    $ExpectedScopeValue = "$ApiAppIdUri/$EntraScopeName"
    $ExpectedCallbackUrl = "https://$ApiPublicHostname/.auth/login/aad/callback"
    $ExpectedAllowedAudiences = Convert-CsvToArray -Value $AllowedAudiences
    $ExpectedAllowedExternalRedirectUrls = @("https://$AppPublicHostname")
    if (-not $AadLoginParametersJson) {
        $ExpectedLoginParameters = Get-DefaultLoginParameters -ScopeValue $ExpectedScopeValue
    } else {
        $ExpectedLoginParameters = $AadLoginParametersJson | ConvertFrom-Json
    }

    Write-Host ">>> Reconciling Entra app registration contract"
    az ad app update `
        --id $EntraAppId `
        --identifier-uris $ApiAppIdUri `
        --web-home-page-url "https://$ApiPublicHostname" `
        --web-redirect-uris $ExpectedCallbackUrl `
        --enable-id-token-issuance true `
        --requested-access-token-version 2 `
        --sign-in-audience AzureADMyOrg `
        --output none | Out-Null

    $currentApi = az rest `
        --method get `
        --url "https://graph.microsoft.com/v1.0/applications/$entraObjectId?`$select=api" `
        --output json | ConvertFrom-Json

    if (-not $currentApi.api) {
        $currentApi | Add-Member -NotePropertyName api -NotePropertyValue ([pscustomobject]@{})
    }
    if (-not $currentApi.api.oauth2PermissionScopes) {
        $currentApi.api | Add-Member -NotePropertyName oauth2PermissionScopes -NotePropertyValue @()
    }

    $scope = $currentApi.api.oauth2PermissionScopes | Where-Object { $_.value -eq $EntraScopeName } | Select-Object -First 1
    if (-not $scope) {
        $scope = [ordered]@{
            id = [guid]::NewGuid().Guid
            value = $EntraScopeName
            type = "User"
            isEnabled = $true
            adminConsentDisplayName = $EntraScopeAdminConsentDisplayName
            adminConsentDescription = $EntraScopeAdminConsentDescription
            userConsentDisplayName = $EntraScopeUserConsentDisplayName
            userConsentDescription = $EntraScopeUserConsentDescription
        }
        $currentApi.api.oauth2PermissionScopes += $scope
    } else {
        $scope.id = if ($scope.id) { $scope.id } else { [guid]::NewGuid().Guid }
        $scope.value = $EntraScopeName
        $scope.type = "User"
        $scope.isEnabled = $true
        $scope.adminConsentDisplayName = $EntraScopeAdminConsentDisplayName
        $scope.adminConsentDescription = $EntraScopeAdminConsentDescription
        $scope.userConsentDisplayName = $EntraScopeUserConsentDisplayName
        $scope.userConsentDescription = $EntraScopeUserConsentDescription
    }

    $currentApi.api.requestedAccessTokenVersion = 2
    $patchBody = @{ api = $currentApi.api } | ConvertTo-Json -Depth 10 -Compress
    az rest `
        --method patch `
        --url "https://graph.microsoft.com/v1.0/applications/$entraObjectId" `
        --body $patchBody `
        --headers "Content-Type=application/json" `
        --output none | Out-Null

    if (-not (Test-AzCommand { az ad sp show --id $EntraAppId --output none 2>$null })) {
        Write-Host ">>> Creating Entra service principal for API app"
        az ad sp create --id $EntraAppId --output none | Out-Null
    }

    if (-not $EntraTenantId) {
        $EntraTenantId = $AccountTenantId
    }
    if (-not $EntraIssuerUrl) {
        $EntraIssuerUrl = "https://login.microsoftonline.com/$EntraTenantId/v2.0"
    }

    if (-not $EntraClientSecret) {
        if ($CreateEntraApp -or $ResetEntraClientSecret) {
            Write-Host ">>> Creating or rotating Entra client secret for Easy Auth"
            $EntraClientSecret = az ad app credential reset `
                --id $EntraAppId `
                --append `
                --display-name $EntraClientSecretDisplayName `
                --query password `
                --output tsv
        } else {
            throw "EntraClientSecret is required unless -CreateEntraApp or -ResetEntraClientSecret is used."
        }
    }

    Upsert-KeyVaultSecret -VaultName $KeyVaultName -SecretName $EntraClientSecretName -SecretValue $EntraClientSecret
}

function Configure-EasyAuth {
    if (-not (Test-ContainerAppExists -Name $ApiAppName)) {
        throw "API app $ApiAppName must exist before Easy Auth can be configured."
    }

    Ensure-EntraAppContract

    Write-Host ">>> Storing Easy Auth client secret on API app"
    az containerapp secret set `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --secrets "$EntraClientSecretName=$EntraClientSecret" `
        --output none | Out-Null

    Write-Host ">>> Configuring Container Apps authentication"
    $loginParametersJson = $ExpectedLoginParameters | ConvertTo-Json -Compress
    az containerapp auth update `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --enabled true `
        --unauthenticated-client-action AllowAnonymous `
        --require-https true `
        --proxy-convention Standard `
        --excluded-paths "/health,/health/readiness" `
        --set "identityProviders.azureActiveDirectory.login.loginParameters=$loginParametersJson" `
        --yes `
        --output none | Out-Null

    $authConfigId = "/subscriptions/$Subscription/resourceGroups/$ResourceGroup/providers/Microsoft.App/containerApps/$ApiAppName/authConfigs/current"
    $auth = az rest `
        --method get `
        --uri "https://management.azure.com$authConfigId?api-version=2025-07-01" `
        --output json | ConvertFrom-Json
    $authProperties = if ($auth.properties) { $auth.properties } else { $auth }
    $loginSettings = @{}
    if ($authProperties.login) {
        foreach ($property in $authProperties.login.PSObject.Properties) {
            $loginSettings[$property.Name] = $property.Value
        }
    }
    $loginSettings["allowedExternalRedirectUrls"] = @($ExpectedAllowedExternalRedirectUrls)
    $authProperties.login = $loginSettings

    $identityProviders = @{}
    if ($authProperties.identityProviders) {
        foreach ($property in $authProperties.identityProviders.PSObject.Properties) {
            $identityProviders[$property.Name] = $property.Value
        }
    }
    $azureActiveDirectory = @{}
    if ($identityProviders.azureActiveDirectory) {
        foreach ($property in $identityProviders.azureActiveDirectory.PSObject.Properties) {
            $azureActiveDirectory[$property.Name] = $property.Value
        }
    }
    $azureActiveDirectory["enabled"] = $true
    $aadLogin = @{}
    if ($azureActiveDirectory.login) {
        foreach ($property in $azureActiveDirectory.login.PSObject.Properties) {
            $aadLogin[$property.Name] = $property.Value
        }
    }
    $aadLogin["loginParameters"] = @($ExpectedLoginParameters)
    $azureActiveDirectory["login"] = $aadLogin
    $aadRegistration = @{}
    if ($azureActiveDirectory.registration) {
        foreach ($property in $azureActiveDirectory.registration.PSObject.Properties) {
            $aadRegistration[$property.Name] = $property.Value
        }
    }
    $aadRegistration["clientId"] = $EntraAppId
    $aadRegistration["clientSecretSettingName"] = $EntraClientSecretName
    $aadRegistration["openIdIssuer"] = $EntraIssuerUrl
    $azureActiveDirectory["registration"] = $aadRegistration
    $aadValidation = @{}
    if ($azureActiveDirectory.validation) {
        foreach ($property in $azureActiveDirectory.validation.PSObject.Properties) {
            $aadValidation[$property.Name] = $property.Value
        }
    }
    $aadValidation["allowedAudiences"] = @($ExpectedAllowedAudiences)
    $azureActiveDirectory["validation"] = $aadValidation
    $identityProviders["azureActiveDirectory"] = $azureActiveDirectory
    $authProperties.identityProviders = $identityProviders

    $authPatchBody = @{ properties = $authProperties } | ConvertTo-Json -Depth 100 -Compress
    az rest `
        --method put `
        --uri "https://management.azure.com$authConfigId?api-version=2025-07-01" `
        --body $authPatchBody `
        --headers "Content-Type=application/json" `
        --output none | Out-Null
}

function Verify-Phase3Contract {
    if (-not $ConfigureEasyAuth) {
        return
    }

    Write-Host ">>> Reading back final auth and ingress state"
    $apiApp = az containerapp show `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --output json | ConvertFrom-Json
    $workerApp = az containerapp show `
        --resource-group $ResourceGroup `
        --name $WorkerAppName `
        --output json | ConvertFrom-Json
    $authConfigId = "/subscriptions/$Subscription/resourceGroups/$ResourceGroup/providers/Microsoft.App/containerApps/$ApiAppName/authConfigs/current"
    $auth = az rest `
        --method get `
        --uri "https://management.azure.com$authConfigId?api-version=2025-07-01" `
        --output json | ConvertFrom-Json
    $auth = if ($auth.properties) { $auth.properties } else { $auth }
    $microsoftAuth = az containerapp auth microsoft show `
        --resource-group $ResourceGroup `
        --name $ApiAppName `
        --output json | ConvertFrom-Json
    $appRegistration = az ad app show `
        --id $EntraAppId `
        --output json | ConvertFrom-Json

    $expectedAppOrigin = "https://$AppPublicHostname"
    $errors = [System.Collections.Generic.List[string]]::new()

    if (-not $auth.platform.enabled) {
        $errors.Add("Easy Auth is not enabled on the API app.")
    }
    if ($auth.globalValidation.unauthenticatedClientAction -ne "AllowAnonymous") {
        $errors.Add("Unexpected unauthenticated action: $($auth.globalValidation.unauthenticatedClientAction)")
    }
    $requireHttps = $auth.httpSettings.requireHttps
    if ($null -eq $requireHttps) {
        $requireHttps = $null -ne $auth.httpSettings.routes.apiPrefix
    }
    if ($requireHttps -ne $true) {
        $errors.Add("Easy Auth HTTPS enforcement is not enabled.")
    }

    $proxyConvention = $auth.httpSettings.forwardProxy.convention
    if ($null -eq $proxyConvention) {
        $proxyConvention = $auth.httpSettings.forwardProxy.proxyConvention
    }
    if ($proxyConvention -ne "Standard") {
        $errors.Add("Unexpected proxy convention: $proxyConvention")
    }

    $excludedPaths = @($auth.globalValidation.excludedPaths)
    if (@("/health", "/health/readiness") -join "," -ne ($excludedPaths | Sort-Object) -join ",") {
        $errors.Add("Unexpected excluded auth paths: $($excludedPaths -join ', ')")
    }

    $actualLoginParameters = @($auth.identityProviders.azureActiveDirectory.login.loginParameters)
    if (($actualLoginParameters -join "|") -ne ($ExpectedLoginParameters -join "|")) {
        $errors.Add("Unexpected login parameters: $($actualLoginParameters -join ', ')")
    }

    $actualAllowedExternalRedirectUrls = @($auth.login.allowedExternalRedirectUrls) | Sort-Object
    if (($actualAllowedExternalRedirectUrls -join "|") -ne (($ExpectedAllowedExternalRedirectUrls | Sort-Object) -join "|")) {
        $errors.Add("Unexpected allowed external redirect URLs: $($actualAllowedExternalRedirectUrls -join ', ')")
    }

    if ($microsoftAuth.registration.clientId -ne $EntraAppId) {
        $errors.Add("Unexpected Entra client ID: $($microsoftAuth.registration.clientId)")
    }
    if ($microsoftAuth.registration.clientSecretSettingName -ne $EntraClientSecretName) {
        $errors.Add("Unexpected Entra client secret setting name: $($microsoftAuth.registration.clientSecretSettingName)")
    }
    if ($microsoftAuth.registration.openIdIssuer -ne $EntraIssuerUrl) {
        $errors.Add("Unexpected issuer URI: $($microsoftAuth.registration.openIdIssuer)")
    }

    $actualAllowedAudiences = @($microsoftAuth.validation.allowedAudiences) | Sort-Object
    $expectedAllowedAudiences = @($ExpectedAllowedAudiences) | Sort-Object
    if (($actualAllowedAudiences -join "|") -ne ($expectedAllowedAudiences -join "|")) {
        $errors.Add("Unexpected allowed audiences: $($actualAllowedAudiences -join ', ')")
    }

    $redirectUris = @($appRegistration.web.redirectUris)
    if ($redirectUris -notcontains $ExpectedCallbackUrl) {
        $errors.Add("Expected callback URL $ExpectedCallbackUrl was not found in app registration redirect URIs: $($redirectUris -join ', ')")
    }

    if ($apiApp.properties.configuration.ingress.external -ne $false) {
        $errors.Add("API app ingress is not internal-only.")
    }
    if ($apiApp.properties.configuration.ingress.targetPort -ne 8000) {
        $errors.Add("Unexpected API target port: $($apiApp.properties.configuration.ingress.targetPort)")
    }
    $corsPolicy = $apiApp.properties.configuration.ingress.corsPolicy
    $corsAllowedOrigins = @($corsPolicy.allowedOrigins) | Sort-Object
    if (($corsAllowedOrigins -join "|") -ne $expectedAppOrigin) {
        $errors.Add("Ingress CORS allowed origins do not match the expected app origin.")
    }
    if ($corsPolicy.allowCredentials -ne $true) {
        $errors.Add("Ingress CORS allowCredentials is not enabled.")
    }
    $corsAllowedMethods = @($corsPolicy.allowedMethods) | Sort-Object
    if (($corsAllowedMethods -join "|") -ne ((@("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT") | Sort-Object) -join "|")) {
        $errors.Add("Ingress CORS allowed methods do not match the expected browser contract.")
    }
    $corsAllowedHeaders = @($corsPolicy.allowedHeaders) | Sort-Object
    if (($corsAllowedHeaders -join "|") -ne "*") {
        $errors.Add("Ingress CORS allowed headers do not match the expected browser contract.")
    }
    if ($corsPolicy.maxAge -ne 600) {
        $errors.Add("Ingress CORS maxAge is not 600.")
    }
    $workerIngress = $workerApp.properties.configuration.ingress
    if ($null -ne $workerIngress -and $workerIngress.PSObject.Properties.Count -gt 0) {
        $errors.Add("Worker app should not expose ingress.")
    }

    $secretNames = @($apiApp.properties.configuration.secrets | ForEach-Object { $_.name })
    if ($secretNames -notcontains $EntraClientSecretName) {
        $errors.Add("Missing API secret setting $EntraClientSecretName.")
    }

    if ($errors.Count -gt 0) {
        throw ($errors -join [Environment]::NewLine)
    }

    Write-Host ">>> Phase 3 auth contract verified successfully"
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
                "REQUIRE_PLATFORM_AUTH=true"
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

    if ($ConfigureEasyAuth) {
        Assert-EasyAuthHostname
        Configure-EasyAuth
        Verify-Phase3Contract
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
if ($ConfigureEasyAuth) {
    Write-Host "Easy Auth hostname: https://$ApiPublicHostname"
    Write-Host "Entra app id: $EntraAppId"
    Write-Host "Entra audience: $ApiAppIdUri"
}
Write-Host ""
if ($ConfigureEasyAuth) {
    Write-Host "Next validation steps:"
    Write-Host "1. Run scripts/validate-aca-phase3-auth.ps1 with the same environment inputs."
    Write-Host "2. Verify browser login reaches $ExpectedCallbackUrl."
    Write-Host "3. Verify /.auth/me returns identity after login and /api/* returns 401 when unauthenticated."
    Write-Host "4. Verify wrong-audience and staging tokens are rejected before promotion."
    Write-Host "5. Run docs/runbooks/origin-bypass-verification.md and confirm no FastAPI log entry exists for direct-origin probes."
} else {
    Write-Host "Next Cloudflare steps:"
    Write-Host "1. Create a remotely managed tunnel for this environment."
    Write-Host "2. Add public hostnames app.<domain> and api.<domain> to the tunnel."
    Write-Host "3. Point app.<domain> to the frontend private origin in ACA."
    Write-Host "4. Point api.<domain> to the API private origin in ACA."
    Write-Host "5. If Easy Auth depends on the public host, set the origin request host header to api.<domain>."
    Write-Host "6. Keep WAF, rate limiting, cache bypass, and optional X-Edge-Secret injection on api.<domain>."
}
