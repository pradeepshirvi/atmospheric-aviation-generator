"""
Synthetic Atmospheric & Aviation Dataset Generator
Final Year Project Implementation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
from scipy.stats import weibull_min
import warnings
warnings.filterwarnings('ignore')

class AtmosphericPhysicsModel:
    """Physics-based models for atmospheric variables"""
    
    # Standard atmosphere constants
    SEA_LEVEL_PRESSURE = 1013.25  # hPa
    SEA_LEVEL_TEMP = 15.0  # Celsius
    TEMP_LAPSE_RATE = -6.5  # Celsius per 1000m
    GAS_CONSTANT = 287.05  # J/(kg·K)
    GRAVITY = 9.80665  # m/s²
    
    @staticmethod
    def calculate_temperature(altitude_m: float, surface_temp: float = None) -> float:
        """Calculate temperature at altitude using ISA model"""
        if surface_temp is None:
            surface_temp = AtmosphericPhysicsModel.SEA_LEVEL_TEMP
        
        # Temperature decreases with altitude (troposphere)
        if altitude_m <= 11000:  # Troposphere
            temp = surface_temp + (AtmosphericPhysicsModel.TEMP_LAPSE_RATE * altitude_m / 1000)
        else:  # Stratosphere (constant temp)
            temp = surface_temp + (AtmosphericPhysicsModel.TEMP_LAPSE_RATE * 11)
        
        return temp
    
    @staticmethod
    def calculate_pressure(altitude_m: float, surface_pressure: float = None) -> float:
        """Calculate pressure at altitude using barometric formula"""
        if surface_pressure is None:
            surface_pressure = AtmosphericPhysicsModel.SEA_LEVEL_PRESSURE
        
        # Barometric formula
        temp_kelvin = AtmosphericPhysicsModel.SEA_LEVEL_TEMP + 273.15
        pressure = surface_pressure * (1 - 0.0065 * altitude_m / temp_kelvin) ** 5.255
        
        return pressure
    
    @staticmethod
    def calculate_humidity(altitude_m: float, surface_humidity: float = 70) -> float:
        """Calculate relative humidity at altitude"""
        # Humidity generally decreases with altitude
        if altitude_m <= 2000:
            humidity = surface_humidity * (1 - altitude_m / 10000)
        else:
            humidity = surface_humidity * 0.8 * np.exp(-altitude_m / 8000)
        
        return max(5, min(100, humidity))  # Clamp between 5-100%

class WindModel:
    """Wind speed and direction modeling"""
    
    @staticmethod
    def generate_wind_profile(altitudes: np.ndarray, 
                            surface_speed: float = 5.0,
                            jet_stream_alt: float = 10000) -> Dict:
        """Generate realistic wind speed and direction profiles"""
        
        wind_speeds = []
        wind_directions = []
        
        for alt in altitudes:
            # Wind speed increases with altitude, peaks at jet stream
            if alt < 1000:
                speed = surface_speed + np.random.normal(0, 1)
            elif alt < jet_stream_alt:
                speed = surface_speed + (alt / jet_stream_alt) * 50 + np.random.normal(0, 5)
            else:
                # Jet stream region
                speed = 50 + 30 * np.exp(-(alt - jet_stream_alt)**2 / 5000000) + np.random.normal(0, 10)
            
            wind_speeds.append(max(0, speed))
            
            # Wind direction varies with altitude (Ekman spiral effect)
            base_direction = 270  # Westerly
            direction = base_direction + 30 * np.sin(alt / 5000) + np.random.normal(0, 20)
            wind_directions.append(direction % 360)
        
        return {
            'speed': np.array(wind_speeds),
            'direction': np.array(wind_directions)
        }

class AviationModel:
    """Aviation-specific parameter generation"""
    
    @staticmethod
    def generate_flight_profile(duration_hours: float = 2.0,
                               cruise_altitude: float = 10000,
                               cruise_speed: float = 250) -> pd.DataFrame:
        """Generate realistic flight profile data"""
        
        # Time points (every 30 seconds)
        time_points = int(duration_hours * 3600 / 30)
        timestamps = pd.date_range(start=datetime.now(), periods=time_points, freq='30S')
        
        # Flight phases: climb, cruise, descent
        climb_time = int(time_points * 0.2)
        descent_time = int(time_points * 0.2)
        cruise_time = time_points - climb_time - descent_time
        
        # Altitude profile
        altitudes = np.concatenate([
            np.linspace(0, cruise_altitude, climb_time),
            np.full(cruise_time, cruise_altitude),
            np.linspace(cruise_altitude, 0, descent_time)
        ])
        
        # Airspeed profile
        airspeeds = np.concatenate([
            np.linspace(0, cruise_speed, climb_time),
            np.full(cruise_time, cruise_speed) + np.random.normal(0, 5, cruise_time),
            np.linspace(cruise_speed, 0, descent_time)
        ])
        
        # Engine parameters
        thrust_percent = np.concatenate([
            np.linspace(100, 75, climb_time),  # High thrust for climb
            np.full(cruise_time, 65) + np.random.normal(0, 5, cruise_time),  # Cruise thrust
            np.linspace(65, 30, descent_time)  # Reduced thrust for descent
        ])
        
        # Fuel flow (kg/hr) - proportional to thrust
        fuel_flow = thrust_percent * 50 + np.random.normal(0, 50, time_points)
        
        # Heading (degrees)
        heading = np.cumsum(np.random.normal(0, 0.5, time_points)) % 360
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'altitude_m': altitudes,
            'airspeed_mps': airspeeds,
            'thrust_percent': thrust_percent,
            'fuel_flow_kg_hr': np.maximum(0, fuel_flow),
            'heading_deg': heading
        })

class SyntheticDataGenerator:
    """Main class for generating synthetic atmospheric and aviation datasets"""
    
    def __init__(self):
        self.atmo_model = AtmosphericPhysicsModel()
        self.wind_model = WindModel()
        self.aviation_model = AviationModel()
    
    def generate_radiosonde_profile(self, 
                                   max_altitude: float = 30000,
                                   num_points: int = 100,
                                   launch_time: datetime = None,
                                   surface_conditions: Dict = None) -> pd.DataFrame:
        """Generate synthetic radiosonde balloon data"""
        
        if launch_time is None:
            launch_time = datetime.now()
        
        if surface_conditions is None:
            surface_conditions = {
                'temperature': 15 + np.random.normal(0, 5),
                'pressure': 1013.25 + np.random.normal(0, 10),
                'humidity': 70 + np.random.normal(0, 15),
                'wind_speed': 5 + np.random.normal(0, 2)
            }
        
        # Generate altitude points
        altitudes = np.linspace(0, max_altitude, num_points)
        
        # Generate atmospheric variables
        temperatures = [self.atmo_model.calculate_temperature(alt, surface_conditions['temperature']) 
                       + np.random.normal(0, 0.5) for alt in altitudes]
        
        pressures = [self.atmo_model.calculate_pressure(alt, surface_conditions['pressure'])
                    + np.random.normal(0, 0.5) for alt in altitudes]
        
        humidity = [self.atmo_model.calculate_humidity(alt, surface_conditions['humidity'])
                   + np.random.normal(0, 2) for alt in altitudes]
        
        # Generate wind profile
        wind_data = self.wind_model.generate_wind_profile(altitudes, surface_conditions['wind_speed'])
        
        # Generate GPS coordinates (simulating balloon drift)
        lat_base = 40.0  # Starting latitude
        lon_base = -105.0  # Starting longitude
        
        # Balloon drifts with wind
        drift_distance = np.cumsum(wind_data['speed'] * 30 / 1000)  # km
        drift_angle = np.radians(wind_data['direction'])
        
        latitudes = lat_base + (drift_distance * np.cos(drift_angle)) / 111  # 1 degree lat = 111 km
        longitudes = lon_base + (drift_distance * np.sin(drift_angle)) / (111 * np.cos(np.radians(lat_base)))
        
        # Time stamps (balloon rises at ~5 m/s)
        ascent_rate = 5.0  # m/s
        time_elapsed = altitudes / ascent_rate
        timestamps = [launch_time + timedelta(seconds=t) for t in time_elapsed]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'altitude_m': altitudes,
            'temperature_c': temperatures,
            'pressure_hpa': pressures,
            'humidity_percent': humidity,
            'wind_speed_mps': wind_data['speed'],
            'wind_direction_deg': wind_data['direction'],
            'latitude': latitudes,
            'longitude': longitudes
        })
    
    def generate_combined_dataset(self,
                                 num_profiles: int = 10,
                                 include_aviation: bool = True) -> Dict:
        """Generate multiple atmospheric profiles with optional aviation data"""
        
        all_radiosonde_data = []
        all_aviation_data = []
        
        for i in range(num_profiles):
            # Vary launch times and conditions
            launch_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
            
            # Generate seasonal variations
            month = launch_time.month
            if month in [12, 1, 2]:  # Winter
                temp_offset = -10
            elif month in [6, 7, 8]:  # Summer
                temp_offset = 10
            else:  # Spring/Fall
                temp_offset = 0
            
            surface_conditions = {
                'temperature': 15 + temp_offset + np.random.normal(0, 5),
                'pressure': 1013.25 + np.random.normal(0, 15),
                'humidity': 60 + np.random.normal(0, 20),
                'wind_speed': 5 + np.random.normal(0, 3)
            }
            
            # Generate radiosonde profile
            radiosonde_data = self.generate_radiosonde_profile(
                max_altitude=25000 + np.random.randint(-5000, 5000),
                num_points=100,
                launch_time=launch_time,
                surface_conditions=surface_conditions
            )
            radiosonde_data['profile_id'] = f'RS_{i:03d}'
            all_radiosonde_data.append(radiosonde_data)
            
            # Generate corresponding aviation data
            if include_aviation:
                flight_data = self.aviation_model.generate_flight_profile(
                    duration_hours=np.random.uniform(1, 4),
                    cruise_altitude=np.random.uniform(8000, 12000),
                    cruise_speed=np.random.uniform(200, 300)
                )
                flight_data['flight_id'] = f'FL_{i:03d}'
                
                # Add atmospheric conditions at flight altitude
                for _, row in flight_data.iterrows():
                    alt = row['altitude_m']
                    flight_data.loc[_, 'ambient_temp_c'] = self.atmo_model.calculate_temperature(alt) + np.random.normal(0, 1)
                    flight_data.loc[_, 'ambient_pressure_hpa'] = self.atmo_model.calculate_pressure(alt)
                
                all_aviation_data.append(flight_data)
        
        # Combine all data
        combined_radiosonde = pd.concat(all_radiosonde_data, ignore_index=True)
        
        result = {
            'radiosonde_data': combined_radiosonde,
            'metadata': {
                'num_profiles': num_profiles,
                'generation_time': datetime.now().isoformat(),
                'altitude_range': [0, combined_radiosonde['altitude_m'].max()],
                'time_range': [
                    combined_radiosonde['timestamp'].min().isoformat(),
                    combined_radiosonde['timestamp'].max().isoformat()
                ]
            }
        }
        
        if include_aviation:
            combined_aviation = pd.concat(all_aviation_data, ignore_index=True)
            result['aviation_data'] = combined_aviation
        
        return result
    
    def export_to_csv(self, data: Dict, output_dir: str = './output'):
        """Export generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export radiosonde data
        if 'radiosonde_data' in data:
            data['radiosonde_data'].to_csv(
                f"{output_dir}/synthetic_radiosonde_data.csv", 
                index=False
            )
            print(f"Radiosonde data exported to {output_dir}/synthetic_radiosonde_data.csv")
        
        # Export aviation data
        if 'aviation_data' in data:
            data['aviation_data'].to_csv(
                f"{output_dir}/synthetic_aviation_data.csv",
                index=False
            )
            print(f"Aviation data exported to {output_dir}/synthetic_aviation_data.csv")
        
        # Export metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(data['metadata'], f, indent=2)
        print(f"Metadata exported to {output_dir}/metadata.json")

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Generate a single radiosonde profile
    print("Generating single radiosonde profile...")
    single_profile = generator.generate_radiosonde_profile()
    print(f"Generated {len(single_profile)} data points")
    print("\nFirst 5 rows:")
    print(single_profile.head())
    
    # Generate combined dataset
    print("\n" + "="*50)
    print("Generating combined dataset with 5 profiles...")
    combined_data = generator.generate_combined_dataset(num_profiles=5, include_aviation=True)
    
    # Export to CSV
    generator.export_to_csv(combined_data)
    
    # Display statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print(f"Total radiosonde records: {len(combined_data['radiosonde_data'])}")
    if 'aviation_data' in combined_data:
        print(f"Total aviation records: {len(combined_data['aviation_data'])}")
    print(f"Altitude range: {combined_data['metadata']['altitude_range']} meters")
    print(f"Time range: {combined_data['metadata']['time_range']}")