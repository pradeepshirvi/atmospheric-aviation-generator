# Docker Deployment Troubleshooting Guide

## Current Issue: Network Connectivity
Docker is unable to pull base images from Docker Hub due to network connectivity issues.

## Solutions:

### Option 1: Fix Docker Network Settings
1. Open Docker Desktop
2. Go to Settings → Resources → Network
3. Try these changes:
   - Disable "Use kernel networking for DNS resolution"
   - Set manual DNS servers: 8.8.8.8, 8.8.4.4
   - Restart Docker Desktop

### Option 2: Use Docker Desktop Registry Mirror
1. In Docker Desktop Settings → Docker Engine
2. Add registry mirrors:
```json
{
  "registry-mirrors": [
    "https://mirror.gcr.io",
    "https://daocloud.io",
    "https://docker.mirrors.ustc.edu.cn"
  ]
}
```

### Option 3: Alternative Deployment Methods

#### A. Run Locally Without Docker
```bash
# Backend
cd D:\python\atmospheric-aviation-generator
pip install -r requirements.txt
python flask_api.py

# Frontend (in new terminal)
npm install
npm run dev
```

#### B. Use Docker Desktop's Built-in Registry
If you have cached images, you can build from scratch using local Python/Node.js

### Option 4: Manual Image Building
Try building with offline/cached images if available.

## Test Commands After Fix:
```bash
# Test connectivity
docker run hello-world

# Build your project
docker-compose up --build -d

# Check running containers
docker ps

# View logs
docker-compose logs
```
