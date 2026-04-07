#!/usr/bin/env bash
# deployment/deploy_azure.sh — Manual Azure deployment script
# Provisions infrastructure and deploys the Docker container to Azure App Service.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker installed and running
#   - .env file configured
#
# Usage:
#   chmod +x deployment/deploy_azure.sh
#   ./deployment/deploy_azure.sh

set -euo pipefail

# ── Configuration — edit these ────────────────────────────────────────────────
RESOURCE_GROUP="acme-rag-rg"
LOCATION="eastus"
ACR_NAME="acmeragacr$(openssl rand -hex 4)"   # Must be globally unique
APP_SERVICE_NAME="acme-rag-agent-$(openssl rand -hex 4)"
OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"     # Set in your environment or .env

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: AZURE_OPENAI_API_KEY environment variable is not set."
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ACME RAG Agent — Azure Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Resource Group : $RESOURCE_GROUP"
echo "Location       : $LOCATION"
echo "ACR Name       : $ACR_NAME"
echo "App Service    : $APP_SERVICE_NAME"
echo ""

# ── Step 1: Create Resource Group ─────────────────────────────────────────────
echo "[1/7] Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# ── Step 2: Deploy Bicep Infrastructure ───────────────────────────────────────
echo "[2/7] Deploying Azure infrastructure (Bicep)..."
DEPLOY_OUTPUT=$(az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --template-file deployment/main.bicep \
  --parameters acrName="$ACR_NAME" appServiceName="$APP_SERVICE_NAME" \
  --query "properties.outputs" -o json)

ACR_LOGIN_SERVER=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['acrLoginServer']['value'])")
APP_URL=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['appServiceUrl']['value'])")
OPENAI_ENDPOINT=$(echo "$DEPLOY_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['openAIEndpoint']['value'])")

echo "  ACR: $ACR_LOGIN_SERVER"
echo "  App URL: $APP_URL"

# ── Step 3: Build Docker image locally ────────────────────────────────────────
echo "[3/7] Building Docker image..."
docker build -t "$ACR_LOGIN_SERVER/rag-agent:latest" .

# ── Step 4: Push to ACR ───────────────────────────────────────────────────────
echo "[4/7] Pushing image to Azure Container Registry..."
az acr login --name "$ACR_NAME"
docker push "$ACR_LOGIN_SERVER/rag-agent:latest"

# ── Step 5: Set Azure OpenAI API Key as App Setting ───────────────────────────
echo "[5/7] Configuring App Service secrets..."
az webapp config appsettings set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_SERVICE_NAME" \
  --settings "AZURE_OPENAI_API_KEY=$OPENAI_API_KEY" \
  --output none

# ── Step 6: Restart App Service ───────────────────────────────────────────────
echo "[6/7] Restarting App Service to pull new image..."
az webapp restart --resource-group "$RESOURCE_GROUP" --name "$APP_SERVICE_NAME"

# ── Step 7: Health check ──────────────────────────────────────────────────────
echo "[7/7] Waiting for app to become healthy (up to 3 minutes)..."
for i in {1..18}; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/health" || true)
  if [ "$STATUS" == "200" ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "  App URL    : $APP_URL"
    echo "  API Docs   : $APP_URL/docs"
    echo "  Health     : $APP_URL/health"
    echo ""
    echo "  Test with:"
    echo "  curl -X POST $APP_URL/ask \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"query\": \"How many days of annual leave do I get?\"}'"
    exit 0
  fi
  echo "  Attempt $i/18 — HTTP $STATUS — waiting 10s..."
  sleep 10
done

echo "⚠️  App may still be starting. Check logs with:"
echo "  az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_SERVICE_NAME"
