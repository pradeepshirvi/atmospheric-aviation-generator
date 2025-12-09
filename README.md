# 🌍✈️ Atmospheric Aviation Dataset Generator

> Generate synthetic atmospheric and aviation data for research, simulation, and machine learning applications.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![React](https://img.shields.io/badge/React-18+-61dafb.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)

## 🚀 Features

- **🌡️ Physics-Based Generation**: Realistic atmospheric profiles using ISA models
- **✈️ Aviation Data**: Flight profiles with engine parameters and atmospheric conditions
- **🤖 ML-Enhanced**: GAN models for advanced synthetic data generation
- **📊 Interactive Dashboard**: Beautiful React frontend with real-time visualization
- **🔬 Data Validation**: Physics-based consistency checks
- **📥 Export Capabilities**: Download data as CSV or JSON
- **🐳 Docker Ready**: One-command deployment
- **☁️ Cloud Deployable**: Ready for Heroku, Vercel, Railway, and more

## 🖼️ Screenshots

### Dashboard Interface
*Interactive dashboard for generating and visualizing atmospheric data*

### Data Visualization  
*Real-time charts showing temperature, pressure, humidity, and wind profiles*

## 🏃‍♂️ Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/yourusername/atmospheric-aviation-generator
cd atmospheric-aviation-generator
docker-compose up --build
```

Visit:
- **Dashboard**: http://localhost
- **API**: http://localhost:5000

### Option 2: Manual Setup

**Backend Setup:**
```bash
pip install -r requirements.txt
python flask_api.py
```

**Frontend Setup:**
```bash
npm install
npm run dev
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/api/generate/radiosonde` | POST | Generate atmospheric profiles |
| `/api/generate/aviation` | POST | Generate flight data |
| `/api/generate/combined` | POST | Generate combined datasets |
| `/api/download` | POST | Download generated data |
| `/api/statistics` | POST | Get data statistics |
| `/api/presets` | GET | Available presets |
| `/api/validate` | POST | Validate data consistency |

### Example API Usage

**Generate Radiosonde Data:**
```bash
curl -X POST http://localhost:5000/api/generate/radiosonde \
  -H "Content-Type: application/json" \
  -d '{
    "min_altitude": 0,
    "max_altitude": 20000,
    "num_points": 100,
    "surface_temp": 15,
    "surface_pressure": 1013.25,
    "surface_humidity": 70
  }'
```

**Generate Aviation Data:**
```bash
curl -X POST http://localhost:5000/api/generate/aviation \
  -H "Content-Type: application/json" \
  -d '{
    "duration_minutes": 120,
    "cruise_altitude": 10000,
    "cruise_speed": 250
  }'
```

## 📊 Data Types

### Atmospheric Data (Radiosonde)
- **Altitude** (meters)
- **Temperature** (°C) 
- **Pressure** (hPa)
- **Humidity** (%)
- **Wind Speed** (m/s)
- **Wind Direction** (degrees)
- **GPS Coordinates** (lat/lon)

### Aviation Data
- **Altitude Profile** (meters)
- **Airspeed** (m/s)
- **Thrust Settings** (%)
- **Fuel Flow** (kg/hr)
- **Heading** (degrees)
- **Ambient Conditions** (temp/pressure)

## 🧪 Machine Learning Models

### GAN (Generative Adversarial Network)
- Generates realistic atmospheric patterns
- Trained on physics-based synthetic data
- Configurable latent dimensions

### VAE (Variational Autoencoder)
- Time-series atmospheric data generation
- LSTM-based encoder/decoder architecture
- Temporal consistency preservation

### Hybrid Physics-ML Approach
- Combines ML generation with physics constraints
- Ensures physical validity
- Customizable constraint parameters

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment guides covering:
- 🐳 Docker deployment
- ☁️ Cloud platforms (Heroku, Vercel, Railway, Render)
- 🌐 Custom domain setup
- 🔒 Security configurations
- 📊 Monitoring and maintenance

### Quick Deploy Options

**Heroku:**
```bash
heroku create your-app-name
git push heroku main
```

**Docker:**
```bash
docker-compose up --build
```

## 🛠️ Development

### Project Structure
```
atmospheric-aviation-generator/
├── 🐍 Backend (Flask)
│   ├── flask_api.py              # Main API server
│   ├── synthetic_generator.py    # Core data generation
│   ├── gan-model.py             # ML models
│   └── requirements.txt         # Python dependencies
├── ⚛️ Frontend (React + TypeScript)
│   ├── src/
│   │   ├── Dashboard.tsx        # Main dashboard component
│   │   ├── App.tsx             # App component
│   │   └── main.tsx            # Entry point
│   ├── package.json            # Node.js dependencies
│   └── vite.config.ts          # Build configuration
└── 🐳 Deployment
    ├── docker-compose.yml      # Full-stack deployment
    ├── Dockerfile.backend      # Backend container
    └── Dockerfile.frontend     # Frontend container
```

### Tech Stack
- **Backend**: Python, Flask, NumPy, Pandas, TensorFlow, Scikit-learn
- **Frontend**: React, TypeScript, Tailwind CSS, Recharts, Vite
- **ML**: TensorFlow/Keras, GANs, VAEs
- **Deployment**: Docker, Docker Compose
- **Cloud**: Heroku, Vercel, Railway, Render

## 📈 Performance

- **Generation Speed**: 1000+ data points/second
- **Memory Usage**: Optimized for large datasets
- **Physics Validation**: Real-time consistency checks
- **Scalability**: Containerized for easy scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- International Standard Atmosphere (ISA) models
- Physics-based atmospheric modeling principles
- Open source machine learning community
- React and Flask communities

## 📞 Support

- 📧 Email: your-email@domain.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/atmospheric-aviation-generator/issues)
- 📚 Documentation: [Full Docs](./DEPLOYMENT.md)

---

**Ready to generate atmospheric data? Start with `docker-compose up --build`! 🚀**
