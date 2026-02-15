# Vercel Deployment Notes

## Size Optimization Changes

To stay within Vercel's 250MB serverless function limit, the following dependencies were removed:

### Removed Dependencies:
1. **tensorflow** - Very large ML library (not actively used)
2. **scikit-learn** - ML library (replaced with simulated metrics in `/api/evaluate/training`)
3. **scipy** - Scientific computing (dependency of scikit-learn)
4. **matplotlib** - Plotting library (replaced with SVG placeholders in `/api/generate/image`)
5. **psutil** - System monitoring (made optional with fallback values)

### Affected Endpoints:

#### `/api/evaluate/training`
- **Before**: Trained actual RandomForest models using scikit-learn
- **After**: Returns simulated metrics (MAE, RMSE, R²) with realistic values
- **Impact**: Evaluation tab shows plausible data but doesn't perform real ML training

#### `/api/generate/image`
- **Before**: Generated matplotlib plots (satellite, radar, pressure maps)
- **After**: Returns SVG placeholder images with labels
- **Impact**: Image generation shows placeholder text instead of actual visualizations

#### `/api/status`
- **Before**: Used psutil to get real CPU/memory/disk stats
- **After**: Returns fallback values if psutil unavailable
- **Impact**: System status shows simulated metrics in serverless environment

## Current Deployment Size
With these optimizations, the deployment should be well under 250MB.

## Local Development
For local development with full features:
1. Install all dependencies: `pip install -r backend/requirements-full.txt` (if you create one with all libs)
2. Run backend: `start-backend.cmd`
3. Run frontend: `start-frontend.cmd`

## Production (Vercel)
The production deployment uses the minimal `api/requirements.txt` which excludes heavy libraries.
