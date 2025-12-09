"""
Flask API Backend for Synthetic Atmospheric & Aviation Dataset Generator
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import zipfile
from typing import Dict, Optional

# Import the generator module (save previous code as synthetic_generator.py)
# For this example, I'll include the essential classes inline

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# =============================================================================
# Core Models (condensed version from main generator)
# =============================================================================

class AtmosphericPhysicsModel:
    """Physics-based models for atmospheric variables"""
    
    @staticmethod
    def calculate_temperature(altitude_m: float, surface_temp: float = 15.0) -> float:
        """Calculate temperature at altitude using ISA model"""
        if altitude_m <= 11000:
            temp = surface_temp + (-6.5 * altitude_m / 1000)
        else:
            temp = surface_temp + (-6.5 * 11)
        return temp
    
    @staticmethod
    def calculate_pressure(altitude_m: float, surface_pressure: float = 1013.25) -> float:
        """Calculate pressure at altitude using barometric formula"""
        temp_kelvin = 288.15
        pressure = surface_pressure * (1 - 0.0065 * altitude_m / temp_kelvin) ** 5.255
        return pressure
    
    @staticmethod
    def calculate_humidity(altitude_m: float, surface_humidity: float = 70) -> float:
        """Calculate relative humidity at altitude"""
        if altitude_m <= 2000:
            humidity = surface_humidity * (1 - altitude_m / 10000)
        else:
            humidity = surface_humidity * 0.8 * np.exp(-altitude_m / 8000)
        return max(5, min(100, humidity))

class QuickDataGenerator:
    """Simplified data generator for API responses"""
    
    def __init__(self):
        self.atmo_model = AtmosphericPhysicsModel()
    
    def generate_quick_profile(self, params: Dict) -> pd.DataFrame:
        """Generate atmospheric profile based on parameters"""
        
        # Extract parameters
        min_alt = params.get('min_altitude', 0)
        max_alt = params.get('max_altitude', 20000)
        num_points = params.get('num_points', 100)
        surface_temp = params.get('surface_temp', 15)
        surface_pressure = params.get('surface_pressure', 1013.25)
        surface_humidity = params.get('surface_humidity', 70)
        
        # Generate altitude points
        altitudes = np.linspace(min_alt, max_alt, num_points)
        
        # Generate data with some random variation
        data = []
        base_time = datetime.now()
        
        for i, alt in enumerate(altitudes):
            temp = self.atmo_model.calculate_temperature(alt, surface_temp) + np.random.normal(0, 0.5)
            pressure = self.atmo_model.calculate_pressure(alt, surface_pressure) + np.random.normal(0, 0.5)
            humidity = self.atmo_model.calculate_humidity(alt, surface_humidity) + np.random.normal(0, 2)
            
            # Wind increases with altitude
            wind_speed = 5 + (alt / 1000) * 2 + np.random.normal(0, 2)
            wind_direction = 270 + np.random.normal(0, 30)
            
            data.append({
                'timestamp': (base_time + timedelta(seconds=i*30)).isoformat(),
                'altitude_m': float(alt),
                'temperature_c': round(temp, 2),
                'pressure_hpa': round(pressure, 2),
                'humidity_percent': round(humidity, 2),
                'wind_speed_mps': round(max(0, wind_speed), 2),
                'wind_direction_deg': round(wind_direction % 360, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_aviation_profile(self, params: Dict) -> pd.DataFrame:
        """Generate aviation data based on parameters"""
        
        duration_min = params.get('duration_minutes', 120)
        cruise_alt = params.get('cruise_altitude', 10000)
        cruise_speed = params.get('cruise_speed', 250)
        
        # Generate time points every 30 seconds
        num_points = duration_min * 2
        
        # Flight phases
        climb_points = int(num_points * 0.2)
        descent_points = int(num_points * 0.2)
        cruise_points = num_points - climb_points - descent_points
        
        # Generate profiles
        altitudes = np.concatenate([
            np.linspace(0, cruise_alt, climb_points),
            np.full(cruise_points, cruise_alt) + np.random.normal(0, 50, cruise_points),
            np.linspace(cruise_alt, 0, descent_points)
        ])
        
        airspeeds = np.concatenate([
            np.linspace(0, cruise_speed, climb_points),
            np.full(cruise_points, cruise_speed) + np.random.normal(0, 5, cruise_points),
            np.linspace(cruise_speed, 0, descent_points)
        ])
        
        thrust = np.concatenate([
            np.linspace(100, 75, climb_points),
            np.full(cruise_points, 65) + np.random.normal(0, 5, cruise_points),
            np.linspace(65, 30, descent_points)
        ])
        
        data = []
        base_time = datetime.now()
        
        for i in range(num_points):
            data.append({
                'timestamp': (base_time + timedelta(seconds=i*30)).isoformat(),
                'altitude_m': round(float(altitudes[i]), 2),
                'airspeed_mps': round(float(airspeeds[i]), 2),
                'thrust_percent': round(float(thrust[i]), 2),
                'fuel_flow_kg_hr': round(max(0, thrust[i] * 50 + np.random.normal(0, 50)), 2),
                'heading_deg': round(np.random.uniform(0, 360), 2),
                'ambient_temp_c': round(self.atmo_model.calculate_temperature(altitudes[i]), 2),
                'ambient_pressure_hpa': round(self.atmo_model.calculate_pressure(altitudes[i]), 2)
            })
        
        return pd.DataFrame(data)

# Initialize generator
generator = QuickDataGenerator()

# =============================================================================
# API Routes
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        'project': 'Synthetic Atmospheric & Aviation Dataset Generator',
        'version': '1.0.0',
        'endpoints': {
            '/api/generate/radiosonde': 'Generate synthetic radiosonde data',
            '/api/generate/aviation': 'Generate synthetic aviation data',
            '/api/generate/combined': 'Generate combined dataset',
            '/api/download': 'Download generated dataset',
            '/api/statistics': 'Get dataset statistics',
            '/api/presets': 'Get available presets'
        }
    })

@app.route('/api/generate/radiosonde', methods=['POST'])
def generate_radiosonde():
    """Generate synthetic radiosonde data"""
    try:
        params = request.json or {}
        
        # Generate data
        df = generator.generate_quick_profile(params)
        
        # Convert to JSON
        result = {
            'success': True,
            'data': df.to_dict(orient='records'),
            'metadata': {
                'num_records': len(df),
                'altitude_range': [df['altitude_m'].min(), df['altitude_m'].max()],
                'generation_time': datetime.now().isoformat()
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/generate/aviation', methods=['POST'])
def generate_aviation():
    """Generate synthetic aviation data"""
    try:
        params = request.json or {}
        
        # Generate data
        df = generator.generate_aviation_profile(params)
        
        # Convert to JSON
        result = {
            'success': True,
            'data': df.to_dict(orient='records'),
            'metadata': {
                'num_records': len(df),
                'flight_duration_min': params.get('duration_minutes', 120),
                'cruise_altitude': params.get('cruise_altitude', 10000),
                'generation_time': datetime.now().isoformat()
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/generate/combined', methods=['POST'])
def generate_combined():
    """Generate combined atmospheric and aviation dataset"""
    try:
        params = request.json or {}
        num_profiles = params.get('num_profiles', 5)
        
        all_radiosonde = []
        all_aviation = []
        
        for i in range(num_profiles):
            # Generate radiosonde profile
            radio_params = {
                'min_altitude': 0,
                'max_altitude': np.random.uniform(15000, 25000),
                'num_points': 100,
                'surface_temp': 15 + np.random.normal(0, 5),
                'surface_pressure': 1013.25 + np.random.normal(0, 10),
                'surface_humidity': 70 + np.random.normal(0, 15)
            }
            radio_df = generator.generate_quick_profile(radio_params)
            radio_df['profile_id'] = f'RS_{i:03d}'
            all_radiosonde.append(radio_df)
            
            # Generate aviation profile
            aviation_params = {
                'duration_minutes': np.random.uniform(60, 240),
                'cruise_altitude': np.random.uniform(8000, 12000),
                'cruise_speed': np.random.uniform(200, 300)
            }
            aviation_df = generator.generate_aviation_profile(aviation_params)
            aviation_df['flight_id'] = f'FL_{i:03d}'
            all_aviation.append(aviation_df)
        
        # Combine data
        combined_radio = pd.concat(all_radiosonde, ignore_index=True)
        combined_aviation = pd.concat(all_aviation, ignore_index=True)
        
        result = {
            'success': True,
            'radiosonde_data': combined_radio.to_dict(orient='records'),
            'aviation_data': combined_aviation.to_dict(orient='records'),
            'metadata': {
                'num_profiles': num_profiles,
                'total_radiosonde_records': len(combined_radio),
                'total_aviation_records': len(combined_aviation),
                'generation_time': datetime.now().isoformat()
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/download', methods=['POST'])
def download_dataset():
    """Download generated dataset as CSV or JSON"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        dataset_type = data.get('dataset_type', 'radiosonde')
        dataset_content = data.get('data', [])
        
        if not dataset_content:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        df = pd.DataFrame(dataset_content)
        
        if format_type == 'csv':
            # Create CSV file in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            # Create response
            response = app.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment;filename=synthetic_{dataset_type}_data.csv'}
            )
            return response
        
        elif format_type == 'json':
            # Create JSON file
            output = io.BytesIO()
            output.write(json.dumps(dataset_content, indent=2).encode())
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'synthetic_{dataset_type}_data.json'
            )
        
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """Get statistical summary of generated data"""
    try:
        data = request.json.get('data', [])
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        df = pd.DataFrame(data)
        
        # Calculate statistics
        stats = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        result = {
            'success': True,
            'statistics': stats,
            'num_records': len(df),
            'columns': list(df.columns)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get available preset configurations"""
    presets = {
        'weather_conditions': {
            'clear_sky': {
                'surface_temp': 20,
                'surface_pressure': 1013.25,
                'surface_humidity': 40
            },
            'stormy': {
                'surface_temp': 10,
                'surface_pressure': 990,
                'surface_humidity': 85
            },
            'winter': {
                'surface_temp': -5,
                'surface_pressure': 1020,
                'surface_humidity': 60
            },
            'summer': {
                'surface_temp': 30,
                'surface_pressure': 1010,
                'surface_humidity': 75
            }
        },
        'flight_profiles': {
            'short_haul': {
                'duration_minutes': 90,
                'cruise_altitude': 8000,
                'cruise_speed': 220
            },
            'medium_haul': {
                'duration_minutes': 180,
                'cruise_altitude': 10000,
                'cruise_speed': 250
            },
            'long_haul': {
                'duration_minutes': 360,
                'cruise_altitude': 12000,
                'cruise_speed': 280
            }
        },
        'altitude_ranges': {
            'low': {'min_altitude': 0, 'max_altitude': 5000},
            'medium': {'min_altitude': 0, 'max_altitude': 15000},
            'high': {'min_altitude': 0, 'max_altitude': 30000}
        }
    }
    
    return jsonify({
        'success': True,
        'presets': presets
    })

@app.route('/api/validate', methods=['POST'])
def validate_data():
    """Validate generated data for physical consistency"""
    try:
        data = request.json.get('data', [])
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        df = pd.DataFrame(data)
        issues = []
        warnings = []
        
        # Check temperature-altitude relationship
        if 'temperature_c' in df.columns and 'altitude_m' in df.columns:
            temp_gradient = df['temperature_c'].diff() / df['altitude_m'].diff()
            avg_gradient = temp_gradient.mean()
            
            if avg_gradient > 0:
                issues.append('Temperature increases with altitude (physically incorrect)')
            elif abs(avg_gradient) < 0.004:
                warnings.append('Temperature lapse rate is unusually low')
            elif abs(avg_gradient) > 0.01:
                warnings.append('Temperature lapse rate is unusually high')
        
        # Check pressure-altitude relationship
        if 'pressure_hpa' in df.columns and 'altitude_m' in df.columns:
            pressure_gradient = df['pressure_hpa'].diff() / df['altitude_m'].diff()
            if pressure_gradient.mean() > 0:
                issues.append('Pressure increases with altitude (physically incorrect)')
        
        # Check humidity bounds
        if 'humidity_percent' in df.columns:
            if (df['humidity_percent'] < 0).any():
                issues.append('Negative humidity values detected')
            if (df['humidity_percent'] > 100).any():
                issues.append('Humidity values exceed 100%')
        
        # Check wind speed
        if 'wind_speed_mps' in df.columns:
            if (df['wind_speed_mps'] < 0).any():
                issues.append('Negative wind speed detected')
            if (df['wind_speed_mps'] > 200).any():
                warnings.append('Extremely high wind speeds detected (>200 m/s)')
        
        # Aviation-specific checks
        if 'thrust_percent' in df.columns:
            if (df['thrust_percent'] < 0).any() or (df['thrust_percent'] > 100).any():
                issues.append('Thrust percentage outside 0-100% range')
        
        if 'fuel_flow_kg_hr' in df.columns:
            if (df['fuel_flow_kg_hr'] < 0).any():
                issues.append('Negative fuel flow detected')
        
        valid = len(issues) == 0
        
        return jsonify({
            'success': True,
            'valid': valid,
            'issues': issues,
            'warnings': warnings,
            'message': 'Data is physically consistent' if valid else 'Data validation failed'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    import os
    print("Starting Synthetic Atmospheric & Aviation Dataset Generator API...")
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    print(f"Server running on port {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
