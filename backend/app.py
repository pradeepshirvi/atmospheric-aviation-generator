"""
Production-ready Flask App for Atmospheric & Aviation Dataset Generator
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_api import app as api_app

def create_app():
    """Create and configure Flask application for production"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
    
    # CORS configuration for production
    cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    # Register blueprints from flask_api
    app.register_blueprint(api_app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
