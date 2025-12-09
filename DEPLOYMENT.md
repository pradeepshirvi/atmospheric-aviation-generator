# 🚀 Atmospheric Aviation Generator - Deployment Guide

This guide covers multiple deployment options for your full-stack atmospheric aviation data generator.

## 📁 Project Structure
```
atmospheric-aviation-generator/
├── 🐍 Backend (Flask API)
│   ├── flask_api.py           # Main API server
│   ├── synthetic_generator.py # Data generation logic
│   ├── gan-model.py          # ML model
│   └── requirements.txt      # Python dependencies
├── ⚛️ Frontend (React)
│   ├── src/                  # React source code
│   ├── package.json          # Node.js dependencies
│   └── vite.config.ts        # Build configuration
└── 🐳 Deployment Files
    ├── Dockerfile.backend    # Backend containerization
    ├── Dockerfile.frontend   # Frontend containerization
    ├── docker-compose.yml    # Full-stack Docker setup
    └── Various platform configs
```

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- Git (for cloning)

### Quick Start
```bash
# Clone your repository
git clone <your-repo-url>
cd atmospheric-aviation-generator

# Build and start both services
docker-compose up --build

# Access your application
# Frontend: http://localhost (port 80)
# Backend API: http://localhost:5000
```

### Individual Container Deployment
```bash
# Build backend
docker build -f Dockerfile.backend -t atmospheric-api .

# Build frontend
docker build -f Dockerfile.frontend -t atmospheric-frontend .

# Run backend
docker run -p 5000:5000 -e FLASK_ENV=production atmospheric-api

# Run frontend
docker run -p 80:80 atmospheric-frontend
```

## ☁️ Cloud Platform Deployments

### 1. 🔺 Vercel + Railway (Recommended for beginners)

**Frontend on Vercel:**
1. Push code to GitHub
2. Connect GitHub to Vercel
3. Deploy automatically
4. Update `vercel.json` with your backend URL

**Backend on Railway:**
1. Connect GitHub to Railway
2. Deploy the root directory
3. Railway will auto-detect Python and use `Procfile`
4. Set environment variables in Railway dashboard

### 2. 🟣 Heroku (Full-stack)

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create apps
heroku create your-atmospheric-api
heroku create your-atmospheric-frontend

# Deploy backend
git subtree push --prefix=backend heroku main

# Deploy frontend
git subtree push --prefix=frontend heroku main

# Set environment variables
heroku config:set FLASK_ENV=production --app your-atmospheric-api
heroku config:set REACT_APP_API_URL=https://your-atmospheric-api.herokuapp.com --app your-atmospheric-frontend
```

### 3. 🚂 Railway (Full-stack)

**Option A: Monorepo Deployment**
1. Connect GitHub repository to Railway
2. Railway will detect both Python and Node.js
3. Configure build and start commands in Railway dashboard

**Option B: Separate Services**
1. Create two Railway services
2. Deploy backend: Use `railway.json` configuration
3. Deploy frontend: Use npm build process

### 4. 🎯 Render (Full-stack)

1. Connect GitHub to Render
2. Use the `render.yaml` file for configuration
3. Render will automatically deploy both services
4. Update CORS settings with your Render URLs

### 5. ▲ Netlify + Backend Platform

**Frontend on Netlify:**
```bash
# Build for production
npm run build

# Deploy to Netlify
# Upload dist/ folder or connect GitHub
```

## 🔧 Environment Configuration

### Backend Environment Variables
```bash
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5000
SECRET_KEY=your-super-secret-key-here
CORS_ORIGINS=https://yourdomain.com,https://another-domain.com
```

### Frontend Environment Variables
```bash
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_VERSION=1.0.0
```

## 🔒 Security Checklist

- [ ] Set `FLASK_DEBUG=False` in production
- [ ] Use strong `SECRET_KEY`
- [ ] Configure CORS properly
- [ ] Enable HTTPS
- [ ] Set up rate limiting (if needed)
- [ ] Remove debug/development dependencies

## 📊 Monitoring & Maintenance

### Health Checks
- Backend: `GET /` returns API status
- Frontend: Check if app loads correctly

### Logs
- Check platform-specific logging dashboards
- Monitor for errors and performance issues

### Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade
npm update

# Rebuild and redeploy
docker-compose up --build
```

## 🌐 Custom Domain Setup

### Vercel
1. Go to Vercel dashboard → Settings → Domains
2. Add your custom domain
3. Configure DNS records

### Heroku
```bash
heroku domains:add yourdomain.com --app your-app-name
```

### Railway/Render
- Configure custom domains in respective dashboards

## 🚨 Troubleshooting

### Common Issues

**CORS Errors:**
- Update `CORS_ORIGINS` environment variable
- Check if frontend URL matches CORS settings

**Build Failures:**
- Verify all dependencies in requirements.txt/package.json
- Check build logs for specific errors

**Port Issues:**
- Ensure your app uses PORT environment variable
- Default ports: Flask=5000, Frontend=80 or 3000

**Environment Variables:**
- Verify all required env vars are set
- Check platform-specific env var syntax

## 📈 Performance Optimization

### Backend
- Use production WSGI server (gunicorn)
- Enable caching for data generation
- Optimize NumPy/Pandas operations

### Frontend
- Build optimization (already configured in Vite)
- Enable gzip compression (configured in nginx.conf)
- Use CDN for static assets

## 📞 Support

If you encounter issues:
1. Check the logs on your chosen platform
2. Verify environment variables
3. Test API endpoints directly
4. Check CORS configuration

## 🎉 Success!

Once deployed, your atmospheric aviation generator will be accessible worldwide:
- Users can generate synthetic atmospheric data
- Interactive dashboard with real-time visualization
- Download capabilities for research use
- Physics-based validation

**Example URLs:**
- Frontend: `https://your-app-name.vercel.app`
- API: `https://your-api-name.railway.app`

---

*Happy deploying! Your atmospheric data generator is now ready for global use! 🌍✈️*
