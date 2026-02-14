# Kubernetes Pod Deployment Guide for Wav2Vec2 ASR

## Overview
This guide helps you deploy the Wav2Vec2 Hindi ASR model as a Kubernetes Pod that runs as a microservice.

## Files Created

1. **Dockerfile** - Container image definition
2. **asr-pod.yaml** - Kubernetes Pod manifest
3. **pod_service.py** - Service script that monitors input directory and processes audio files
4. **DEPLOYMENT.md** - This file

## Prerequisites

- Kubernetes cluster running (can be local with minikube or on cloud)
- Docker installed and configured
- `kubectl` command-line tool
- Access to your container registry (Docker Hub, ECR, GCR, etc.)

## Step 1: Build the Docker Image

```bash
cd /home/ubuntu/ASR/ASR/Wav2Vec2

# Build the Docker image
docker build -t wav2vec2-asr:latest .

# Optionally tag for your registry
docker tag wav2vec2-asr:latest your-registry/wav2vec2-asr:latest
docker push your-registry/wav2vec2-asr:latest
```

## Step 2: Deploy the Pod

```bash
# Check your current kubernetes context
kubectl config current-context

# Create the pod
kubectl apply -f asr-pod.yaml

# Check pod status
kubectl get pods -l app=wav2vec2-asr

# View pod logs
kubectl logs wav2vec2-asr-pod

# Describe the pod (for troubleshooting)
kubectl describe pod wav2vec2-asr-pod
```

## Step 3: Using the Pod (File Processing)

The pod continuously monitors `/app/input` directory for audio files and writes transcriptions to `/app/output`.

### Copy files to the pod:
```bash
# Copy audio file to pod
kubectl cp your_audio.wav wav2vec2-asr-pod:/app/input/

# Copy entire directory
kubectl cp ./audio_files/ wav2vec2-asr-pod:/app/input/
```

### Retrieve results:
```bash
# Get transcription JSON output
kubectl cp wav2vec2-asr-pod:/app/output/ ./results/

# Or stream logs
kubectl logs -f wav2vec2-asr-pod
```

## Advanced Deployment Options

### Option 1: Using a ConfigMap for files

```bash
# Create configmap with audio files
kubectl create configmap asr-inputs --from-file=./audio_files/

# Use in pod (update asr-pod.yaml volumes section)
```

### Option 2: Using Persistent Volumes

```yaml
volumes:
- name: audio-storage
  persistentVolumeClaim:
    claimName: asr-pvc
```

### Option 3: Deploy as a Service for API Access

Create `asr-service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: wav2vec2-asr-service
spec:
  type: LoadBalancer
  selector:
    app: wav2vec2-asr
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
```

## Kubernetes Pod Commands Reference

```bash
# Create pod
kubectl apply -f asr-pod.yaml

# Check pod status
kubectl get pod wav2vec2-asr-pod
kubectl get pod wav2vec2-asr-pod -o wide
kubectl get pod wav2vec2-asr-pod -o yaml

# View logs
kubectl logs wav2vec2-asr-pod
kubectl logs -f wav2vec2-asr-pod  # Follow logs
kubectl logs --previous wav2vec2-asr-pod  # Previous container logs

# Execute command in pod
kubectl exec -it wav2vec2-asr-pod -- bash
kubectl exec wav2vec2-asr-pod -- ls -la /app/output

# Copy files
kubectl cp local_file wav2vec2-asr-pod:/app/input/
kubectl cp wav2vec2-asr-pod:/app/output/ local_directory/

# Delete pod
kubectl delete pod wav2vec2-asr-pod
kubectl delete -f asr-pod.yaml

# Describe pod (debugging)
kubectl describe pod wav2vec2-asr-pod

# Port forward (for services)
kubectl port-forward wav2vec2-asr-pod 8000:8000
```

## Monitoring and Debugging

### Check resource usage:
```bash
kubectl top pod wav2vec2-asr-pod
kubectl top nodes
```

### View events:
```bash
kubectl get events --sort-by='.lastTimestamp'
```

### Enter the pod:
```bash
kubectl exec -it wav2vec2-asr-pod -- /bin/bash
# Check model file
ls -lh /app/hindi.pt
# Check output directory
ls -la /app/output/
```

## Performance Tips

1. **Node Affinity**: Deploy on nodes with sufficient CPU (preferably Xeon)
2. **Resource Limits**: Adjust memory/CPU in asr-pod.yaml as needed
3. **Batch Processing**: Copy multiple audio files at once
4. **Monitoring**: Use `kubectl logs -f` to monitor in real-time

## Cleanup

```bash
# Delete the pod
kubectl delete -f asr-pod.yaml

# Delete the image
docker rmi wav2vec2-asr:latest
```

## Troubleshooting

### Pod doesn't start
```bash
kubectl describe pod wav2vec2-asr-pod
kubectl logs wav2vec2-asr-pod
```

### Out of memory
Increase memory limit in asr-pod.yaml:
```yaml
resources:
  limits:
    memory: "8Gi"
```

### Model file not found
Ensure `hindi.pt` is in the build context:
```bash
ls -lh /home/ubuntu/ASR/ASR/Wav2Vec2/hindi.pt
```

### Slow inference
Check pod logs for warnings and CPU usage:
```bash
kubectl logs wav2vec2-asr-pod | grep -i error
kubectl top pod wav2vec2-asr-pod
```

## Next Steps

1. Create a Kubernetes Deployment (for multiple replicas)
2. Add ingress for API access
3. Set up monitoring with Prometheus
4. Create a web UI for uploading audio
5. Integrate with message queues (Kafka, RabbitMQ)

## Example: Full Deployment Workflow

```bash
# 1. Build image
docker build -t wav2vec2-asr:latest .

# 2. Create pod
kubectl apply -f asr-pod.yaml

# 3. Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=wav2vec2-asr --timeout=300s

# 4. Copy audio file
kubectl cp test_audio.wav wav2vec2-asr-pod:/app/input/test.wav

# 5. Wait for processing
sleep 10

# 6. Get results
kubectl cp wav2vec2-asr-pod:/app/output/ ./results/

# 7. View transcription
cat results/test_transcription.json
```

---

**Happy Kubernetes deployment!** 🚀
