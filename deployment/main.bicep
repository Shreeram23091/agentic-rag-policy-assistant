// deployment/main.bicep — Azure Infrastructure as Code
// Provisions: Resource Group resources for ACME RAG Agent
// Deploy with: az deployment group create --resource-group <rg> --template-file main.bicep --parameters @params.json

@description('Location for all resources')
param location string = resourceGroup().location

@description('App Service Plan SKU')
param appServicePlanSku string = 'B2'  // Basic tier, 2 cores — upgrade to P2v3 for production

@description('Azure Container Registry name (must be globally unique)')
param acrName string = 'acmeragacr${uniqueString(resourceGroup().id)}'

@description('App Service name (must be globally unique)')
param appServiceName string = 'acme-rag-agent-${uniqueString(resourceGroup().id)}'

@description('Azure OpenAI resource name')
param openAIName string = 'acme-openai-${uniqueString(resourceGroup().id)}'

@description('Container image tag to deploy')
param imageTag string = 'latest'

// ── Azure Container Registry ──────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ── Azure OpenAI ──────────────────────────────────────────────────────────────
resource openAI 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: openAIName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAIName
    publicNetworkAccess: 'Enabled'
  }
}

resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: openAI
  name: 'gpt-4o'
  sku: {
    name: 'Standard'
    capacity: 30  // TPM in thousands
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-05-13'
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: openAI
  name: 'text-embedding-3-small'
  sku: {
    name: 'Standard'
    capacity: 120
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-small'
      version: '1'
    }
  }
  dependsOn: [gpt4oDeployment]
}

// ── App Service Plan ──────────────────────────────────────────────────────────
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: '${appServiceName}-plan'
  location: location
  kind: 'linux'
  sku: {
    name: appServicePlanSku
  }
  properties: {
    reserved: true  // Required for Linux
  }
}

// ── App Service (Web App for Containers) ─────────────────────────────────────
resource appService 'Microsoft.Web/sites@2023-01-01' = {
  name: appServiceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|${acr.properties.loginServer}/rag-agent:${imageTag}'
      alwaysOn: true
      appSettings: [
        { name: 'DOCKER_REGISTRY_SERVER_URL',      value: 'https://${acr.properties.loginServer}' }
        { name: 'DOCKER_REGISTRY_SERVER_USERNAME', value: acr.name }
        { name: 'DOCKER_REGISTRY_SERVER_PASSWORD', value: acr.listCredentials().passwords[0].value }
        { name: 'WEBSITES_PORT',                   value: '8000' }
        { name: 'OPENAI_API_TYPE',                 value: 'azure' }
        { name: 'AZURE_OPENAI_ENDPOINT',           value: openAI.properties.endpoint }
        { name: 'AZURE_OPENAI_API_VERSION',        value: '2024-02-01' }
        { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT',    value: 'gpt-4o' }
        { name: 'AZURE_OPENAI_EMBED_DEPLOYMENT',   value: 'text-embedding-3-small' }
        { name: 'ENVIRONMENT',                     value: 'production' }
        { name: 'FAISS_INDEX_PATH',                value: '/home/data/faiss_index' }
        { name: 'DOCUMENTS_PATH',                  value: '/app/docs/sample_documents' }
        // AZURE_OPENAI_API_KEY must be set manually via Azure Portal or Key Vault reference
        // { name: 'AZURE_OPENAI_API_KEY', value: '...' }
      ]
    }
    httpsOnly: true
  }
}

// ── Application Insights ──────────────────────────────────────────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${appServiceName}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

// ── Outputs ────────────────────────────────────────────────────────────────────
output appServiceUrl string = 'https://${appService.properties.defaultHostName}'
output acrLoginServer string = acr.properties.loginServer
output openAIEndpoint string = openAI.properties.endpoint
output appInsightsConnectionString string = appInsights.properties.ConnectionString
