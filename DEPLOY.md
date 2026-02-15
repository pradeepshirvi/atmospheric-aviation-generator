# Deployment Guide for Atmospheric Aviation Generator on Vercel

This project is configured for deployment on [Vercel](https://vercel.com).

## Prerequisites
- A Vercel account
- GitHub repository for this project

## Steps to Deploy

1.  **Push your code to GitHub**
    - Ensure all changes are committed and pushed to your repository.

2.  **Import Project in Vercel**
    - Go to your Vercel Dashboard.
    - Click "Add New..." -> "Project".
    - Import your GitHub repository.

3.  **Configure Project**
    - **Framework Preset**: Vercel should auto-detect `Vite`.
    - **Root Directory**: `./` (default)
    - **Build Command**: `npm run build` (or `vite build`) - Vercel should auto-detect.
    - **Output Directory**: `dist` (default)
    - **Environment Variables**:
      - Add `FLASK_DEBUG` = `False` (Recommended for production)

4.  **Deploy**
    - Click "Deploy".
    - Vercel will install Python dependencies from `requirements.txt` and build the frontend.

## API Configuration
The project is configured with a `vercel.json` file that routes all `/api/*` requests to the Python backend (`api/index.py`).
The frontend is configured to use relative paths (`/api/...`) which works in production.

## Local Development
To run locally:
1.  Start the backend: `python start-backend.ps1` (or `python flask_api.py`)
2.  Start the frontend: `npm run dev`
3.  Access the app at `http://localhost:3000`. API requests are proxied to port 5000.
