import os
import sys

# Add parent directory to path so we can import flask_api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_api import app

# This is required for Vercel to pick up the app object
# Vercel looks for a variable named 'app'
