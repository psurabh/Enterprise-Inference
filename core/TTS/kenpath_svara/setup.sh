#!/usr/bin/env bash
# =============================================================================
# Svara TTS - Full Stack Setup Script
# =============================================================================
# Usage:
#   git clone https://github.com/psurabh/Enterprise-Inference.git
#   cd Enterprise-Inference/core/TTS/kenpath_svara
#   chmod +x setup.sh && ./setup.sh
#
# What this script does:
#   1. Validates prerequisites (kubectl, helm, git)
#   2. Labels the inference node for pod scheduling
#   3. Creates the HuggingFace token secret (needed to pull kenpath/svara-tts-v1)
#   4. Applies the PVC for VLLM model storage
#   5. Deploys the VLLM pod via Helm (kenpath/svara-tts-v1 on CPU/Xeon)
#   6. Deploys the SNAC Decoder + FastAPI server via kubectl
#   7. Waits for both pods to become Ready
#   8. Prints all live endpoints
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# Logging setup — everything goes to terminal AND a timestamped log file
# --------------------------------------------------------------------------
LOG_DIR="/tmp/svara-tts-setup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
# Tee all stdout+stderr to the log file
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setup log started — writing to $LOG_FILE"

# --------------------------------------------------------------------------
# Colour helpers (colours in terminal; plain text lands in log file via tee)
# --------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ts()      { date '+%Y-%m-%d %H:%M:%S'; }
info()    { echo -e "[$(ts)] ${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "[$(ts)] ${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "[$(ts)] ${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "[$(ts)] ${RED}[ERROR]${NC} $*"; exit 1; }

# --------------------------------------------------------------------------
# Configuration — edit here if needed
# --------------------------------------------------------------------------
REPO_URL="https://github.com/psurabh/Enterprise-Inference.git"
REPO_BRANCH="feat/svara-tts-cpu-integration"
NAMESPACE="default"
INFERENCE_NODE="master1"
NODE_ROLE_LABEL="role=inference"

HELM_RELEASE="svara-cpu-vllm"
HELM_CHART_DIR="../../kubespray/helm-charts/vllm"   # relative to this script
HELM_VALUES_FILE="xeon-values.yaml"
VLLM_MODEL="kenpath/svara-tts-v1"
VLLM_NODEPORT="31257"
VLLM_SERVICE_NAME="svara-cpu-vllm-service"

SNAC_MANIFEST="api/k8s-deployment.yaml"
SNAC_PVC_MANIFEST="api/svara-cpu-vllm-pvc.yaml"
SNAC_DEPLOYMENT="snac-decoder"
SNAC_SERVICE="snac-decoder-service"
SNAC_NODEPORT="30800"

HF_SECRET_NAME="svara-cpu-vllm-secret"

# --------------------------------------------------------------------------
# Resolve script directory (works even when called from another cwd)
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================================"
echo "  Svara TTS – Full Stack Deployment"
echo "========================================================"
echo ""

# --------------------------------------------------------------------------
# Step 0: Prerequisites
# --------------------------------------------------------------------------
info "Checking prerequisites..."
for cmd in kubectl helm git curl; do
    if ! command -v "$cmd" &>/dev/null; then
        error "'$cmd' is not installed or not in PATH."
    fi
done
kubectl cluster-info --request-timeout=5s &>/dev/null || error "Cannot reach Kubernetes cluster. Is kubeconfig set?"
success "All prerequisites satisfied"

# --------------------------------------------------------------------------
# Step 1: Clone repo (skip if already inside it)
# --------------------------------------------------------------------------
info "Checking repository..."
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel &>/dev/null; then
    REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
    success "Already inside repo at: $REPO_ROOT"
else
    warn "Not inside a git repo. Cloning..."
    CLONE_DIR="$(pwd)/Enterprise-Inference"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR"
    cd "$CLONE_DIR/core/TTS/kenpath_svara"
    SCRIPT_DIR="$(pwd)"
    success "Cloned to: $CLONE_DIR"
fi

# --------------------------------------------------------------------------
# Step 2: Label inference node
# --------------------------------------------------------------------------
info "Labelling node '$INFERENCE_NODE' with '$NODE_ROLE_LABEL'..."
if kubectl label node "$INFERENCE_NODE" "$NODE_ROLE_LABEL" --overwrite &>/dev/null; then
    success "Node labelled: $INFERENCE_NODE $NODE_ROLE_LABEL"
else
    warn "Could not label node '$INFERENCE_NODE' (may not exist or already labelled)"
fi

# --------------------------------------------------------------------------
# Step 3: HuggingFace token secret (skipped)
# --------------------------------------------------------------------------
info "Skipping HuggingFace token secret step."
# To enable, create the secret manually:
#   kubectl create secret generic svara-cpu-vllm-secret \
#       --from-literal=HF_TOKEN=<your_token> -n default

# --------------------------------------------------------------------------
# Step 4: PVC for VLLM model storage
# --------------------------------------------------------------------------
info "Applying PVC manifest: $SNAC_PVC_MANIFEST"
if kubectl get pvc svara-cpu-vllm-pvc -n "$NAMESPACE" &>/dev/null; then
    success "PVC 'svara-cpu-vllm-pvc' already exists — skipping"
else
    kubectl apply -f "$SNAC_PVC_MANIFEST" -n "$NAMESPACE"
    success "PVC created"
fi

# --------------------------------------------------------------------------
# Step 5: Deploy VLLM via Helm
# --------------------------------------------------------------------------
CHART_PATH="$SCRIPT_DIR/$HELM_CHART_DIR"
VALUES_PATH="$CHART_PATH/$HELM_VALUES_FILE"

info "Deploying VLLM Helm release '$HELM_RELEASE'..."
if helm status "$HELM_RELEASE" -n "$NAMESPACE" &>/dev/null; then
    success "Helm release '$HELM_RELEASE' already deployed — running upgrade..."
    helm upgrade "$HELM_RELEASE" "$CHART_PATH" \
        -n "$NAMESPACE" \
        -f "$VALUES_PATH" \
        --set LLM_MODEL_ID="$VLLM_MODEL" \
        --set SERVED_MODEL_NAME="$VLLM_MODEL" \
        --set service.nodePort="$VLLM_NODEPORT"
else
    helm install "$HELM_RELEASE" "$CHART_PATH" \
        -n "$NAMESPACE" \
        -f "$VALUES_PATH" \
        --set LLM_MODEL_ID="$VLLM_MODEL" \
        --set SERVED_MODEL_NAME="$VLLM_MODEL" \
        --set service.nodePort="$VLLM_NODEPORT"
    success "Helm release '$HELM_RELEASE' installed"
fi

# --------------------------------------------------------------------------
# Step 6: Deploy SNAC Decoder (FastAPI server)
# --------------------------------------------------------------------------
info "Applying SNAC Decoder manifest: $SNAC_MANIFEST"
kubectl apply -f "$SNAC_MANIFEST" -n "$NAMESPACE"
success "SNAC Decoder manifests applied"

# --------------------------------------------------------------------------
# Step 7: Wait for pods to be Ready
# --------------------------------------------------------------------------
echo ""
info "Waiting for VLLM pod to be Ready (timeout: 15min — model download may take time)..."
kubectl rollout status deployment/"$HELM_RELEASE" -n "$NAMESPACE" --timeout=900s \
    && success "VLLM deployment is Ready" \
    || warn "VLLM deployment timed out — check: kubectl logs -n $NAMESPACE deploy/$HELM_RELEASE"

echo ""
info "Waiting for SNAC Decoder pod to be Ready (timeout: 10min — pip install runs on start)..."
kubectl rollout status deployment/"$SNAC_DEPLOYMENT" -n "$NAMESPACE" --timeout=600s \
    && success "SNAC Decoder deployment is Ready" \
    || warn "SNAC Decoder timed out — check: kubectl logs -n $NAMESPACE deploy/$SNAC_DEPLOYMENT"

# --------------------------------------------------------------------------
# Step 8: Print all endpoints
# --------------------------------------------------------------------------
NODE_IP=$(kubectl get node "$INFERENCE_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null || echo "localhost")

EXTERNAL_IP=$(curl -sf --max-time 3 http://checkip.amazonaws.com 2>/dev/null || echo "$NODE_IP")

echo ""
echo "========================================================"
echo -e "${GREEN}  Svara TTS Stack is UP${NC}"
echo "========================================================"
echo ""
echo -e "${CYAN}  VLLM (Token Generation — kenpath/svara-tts-v1)${NC}"
echo "  ┌─────────────────────────────────────────────────────"
echo "  │  Internal:  http://$NODE_IP:$VLLM_NODEPORT"
echo "  │  Health:    http://$NODE_IP:$VLLM_NODEPORT/health"
echo "  │  Models:    http://$NODE_IP:$VLLM_NODEPORT/v1/models"
echo "  │  K8s Svc:   $VLLM_SERVICE_NAME.$NAMESPACE.svc.cluster.local:80"
echo "  └─────────────────────────────────────────────────────"
echo ""
echo -e "${CYAN}  SNAC Decoder + FastAPI (TTS API Server)${NC}"
echo "  ┌─────────────────────────────────────────────────────"
echo "  │  Internal:  http://$NODE_IP:$SNAC_NODEPORT"
echo "  │  Health:    http://$NODE_IP:$SNAC_NODEPORT/health"
echo "  │  Voices:    http://$NODE_IP:$SNAC_NODEPORT/v1/voices"
echo "  │  TTS:       http://$NODE_IP:$SNAC_NODEPORT/v1/text-to-speech  [POST]"
echo "  │  Docs:      http://$NODE_IP:$SNAC_NODEPORT/docs"
echo "  │  K8s Svc:   $SNAC_SERVICE.$NAMESPACE.svc.cluster.local:8000"
echo "  └─────────────────────────────────────────────────────"
echo ""
echo -e "${CYAN}  Quick Test Commands${NC}"
echo "  ┌─────────────────────────────────────────────────────"
echo "  │  # Health check"
echo "  │  curl http://$NODE_IP:$SNAC_NODEPORT/health"
echo "  │"
echo "  │  # List voices"
echo "  │  curl http://$NODE_IP:$SNAC_NODEPORT/v1/voices | python3 -m json.tool"
echo "  │"
echo "  │  # Generate speech (saves to speech.wav)"
echo "  │  curl -X POST http://$NODE_IP:$SNAC_NODEPORT/v1/text-to-speech \\"
echo "  │       -H 'Content-Type: application/json' \\"
echo "  │       -d '{\"text\": \"Hello, this is Svara TTS.\", \"voice\": \"en_male\"}' \\"
echo "  │       --output speech.wav"
echo "  └─────────────────────────────────────────────────────"
echo ""
echo -e "${GREEN}  Done!${NC} To monitor pods:  kubectl get pods -n $NAMESPACE -w"
echo ""
echo -e "${CYAN}  Full log saved to:${NC} $LOG_FILE"
echo ""
