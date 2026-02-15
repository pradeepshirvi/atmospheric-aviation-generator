import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Download, Cloud, Plane, Activity, Settings, Database, CheckCircle, AlertCircle, Image, BarChart2, Cpu } from 'lucide-react';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('radiosonde');
  const [generatedData, setGeneratedData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState<any>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [generatedImage, setGeneratedImage] = useState<any>(null);
  const [evaluationMetrics, setEvaluationMetrics] = useState<any>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [imageType, setImageType] = useState('satellite');
  const [trainingMetrics, setTrainingMetrics] = useState<any>(null);

  // Form parameters
  const [parameters, setParameters] = useState({
    // Radiosonde parameters
    min_altitude: 0,
    max_altitude: 20000,
    num_points: 100,
    surface_temp: 15,
    surface_pressure: 1013.25,
    surface_humidity: 70,
    // Aviation parameters
    duration_minutes: 120,
    cruise_altitude: 10000,
    cruise_speed: 250,
    // Combined parameters
    num_profiles: 5
  });

  const [selectedPreset, setSelectedPreset] = useState('');

  // Weather presets
  const weatherPresets = {
    clear: { surface_temp: 20, surface_pressure: 1013.25, surface_humidity: 40 },
    stormy: { surface_temp: 10, surface_pressure: 990, surface_humidity: 85 },
    winter: { surface_temp: -5, surface_pressure: 1020, surface_humidity: 60 },
    summer: { surface_temp: 30, surface_pressure: 1010, surface_humidity: 75 }
  };

  // Flight presets
  const flightPresets = {
    short_haul: { duration_minutes: 90, cruise_altitude: 8000, cruise_speed: 220 },
    medium_haul: { duration_minutes: 180, cruise_altitude: 10000, cruise_speed: 250 },
    long_haul: { duration_minutes: 360, cruise_altitude: 12000, cruise_speed: 280 }
  };

  const handleParameterChange = (key, value) => {
    setParameters(prev => ({
      ...prev,
      [key]: parseFloat(value) || value
    }));
  };

  const applyPreset = (type, preset) => {
    if (type === 'weather') {
      setParameters(prev => ({
        ...prev,
        ...weatherPresets[preset]
      }));
    } else if (type === 'flight') {
      setParameters(prev => ({
        ...prev,
        ...flightPresets[preset]
      }));
    }
  };

  const generateImage = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/generate/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: imageType })
      });
      const data = await response.json();
      if (data.success) {
        setGeneratedImage(data);
      }
    } catch (error) {
      console.error('Error generating image:', error);
    }
    setLoading(false);
  };

  const evaluateModel = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/evaluate', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setEvaluationMetrics(data);
      }
    } catch (error) {
      console.error('Error evaluating model:', error);
    }
    setLoading(false);
  };

  const evaluateTraining = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/evaluate/training', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setTrainingMetrics(data);
      }
    } catch (error) {
      console.error('Error evaluating training:', error);
    }
    setLoading(false);
  };

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      if (data.success) {
        setSystemStatus(data);
      }
    } catch (error) {
      console.error('Error checking system status:', error);
    }
  };

  // Poll system status when on that tab
  useEffect(() => {
    if (activeTab === 'status') {
      checkSystemStatus();
      const interval = setInterval(checkSystemStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  const generateData = async () => {
    setLoading(true);
    setValidationResult(null);

    // Simulate API call
    setTimeout(() => {
      let data = [];

      if (activeTab === 'radiosonde') {
        // Generate radiosonde data
        const altitudes = [];
        for (let i = 0; i <= parameters.num_points; i++) {
          const alt = parameters.min_altitude + (parameters.max_altitude - parameters.min_altitude) * i / parameters.num_points;
          altitudes.push(alt);
        }

        data = altitudes.map((alt, i) => ({
          index: i,
          altitude_m: Math.round(alt),
          temperature_c: parseFloat((parameters.surface_temp - 6.5 * alt / 1000 + (Math.random() - 0.5)).toFixed(2)),
          pressure_hpa: parseFloat((parameters.surface_pressure * Math.pow(1 - 0.0065 * alt / 288.15, 5.255)).toFixed(2)),
          humidity_percent: parseFloat(Math.max(5, Math.min(100, parameters.surface_humidity * Math.exp(-alt / 8000) + (Math.random() - 0.5) * 10)).toFixed(2)),
          wind_speed_mps: parseFloat((5 + alt / 1000 * 2 + (Math.random() - 0.5) * 4).toFixed(2)),
          wind_direction_deg: Math.round(270 + (Math.random() - 0.5) * 60)
        }));
      } else if (activeTab === 'aviation') {
        // Generate aviation data
        const numPoints = parameters.duration_minutes * 2; // Every 30 seconds
        const climbPoints = Math.floor(numPoints * 0.2);
        const descentPoints = Math.floor(numPoints * 0.2);
        const cruisePoints = numPoints - climbPoints - descentPoints;

        for (let i = 0; i < numPoints; i++) {
          let altitude, airspeed, thrust;

          if (i < climbPoints) {
            altitude = (parameters.cruise_altitude * i) / climbPoints;
            airspeed = (parameters.cruise_speed * i) / climbPoints;
            thrust = 100 - (25 * i) / climbPoints;
          } else if (i < climbPoints + cruisePoints) {
            altitude = parameters.cruise_altitude;
            airspeed = parameters.cruise_speed + (Math.random() - 0.5) * 10;
            thrust = 65 + (Math.random() - 0.5) * 10;
          } else {
            const descentIndex = i - climbPoints - cruisePoints;
            altitude = parameters.cruise_altitude * (1 - descentIndex / descentPoints);
            airspeed = parameters.cruise_speed * (1 - descentIndex / descentPoints);
            thrust = 65 - (35 * descentIndex) / descentPoints;
          }

          data.push({
            index: i,
            time_min: i * 0.5,
            altitude_m: Math.round(altitude),
            airspeed_mps: Math.round(airspeed),
            thrust_percent: Math.round(thrust),
            fuel_flow_kg_hr: Math.round(thrust * 50 + (Math.random() - 0.5) * 100),
            ambient_temp_c: parseFloat((15 - 6.5 * altitude / 1000).toFixed(2))
          });
        }
      }

      setGeneratedData(data);
      calculateStatistics(data);
      validateData(data);
      setLoading(false);
    }, 1000);
  };

  const calculateStatistics = (data) => {
    if (!data || data.length === 0) return;

    const stats = {};
    const numericKeys = Object.keys(data[0]).filter(key => typeof data[0][key] === 'number' && key !== 'index');

    numericKeys.forEach(key => {
      const values = data.map(d => d[key]);
      stats[key] = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        median: values.sort((a, b) => a - b)[Math.floor(values.length / 2)]
      };
    });

    setStatistics(stats);
  };

  const validateData = (data) => {
    const issues = [];
    const warnings = [];

    if (activeTab === 'radiosonde' && data.length > 1) {
      // Check temperature decrease with altitude
      const tempGradient = (data[data.length - 1].temperature_c - data[0].temperature_c) /
        (data[data.length - 1].altitude_m - data[0].altitude_m);
      if (tempGradient > 0) {
        issues.push('Temperature increases with altitude');
      }

      // Check pressure decrease
      const pressureGradient = (data[data.length - 1].pressure_hpa - data[0].pressure_hpa) /
        (data[data.length - 1].altitude_m - data[0].altitude_m);
      if (pressureGradient > 0) {
        issues.push('Pressure increases with altitude');
      }

      // Check for extreme wind speeds
      const maxWind = Math.max(...data.map(d => d.wind_speed_mps));
      if (maxWind > 150) {
        warnings.push('Very high wind speeds detected (>150 m/s)');
      }
    }

    setValidationResult({
      valid: issues.length === 0,
      issues,
      warnings
    });
  };

  const downloadData = (format) => {
    if (!generatedData) return;

    if (format === 'csv') {
      const headers = Object.keys(generatedData[0]).join(',');
      const rows = generatedData.map(row => Object.values(row).join(','));
      const csv = [headers, ...rows].join('\n');

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `synthetic_${activeTab}_data.csv`;
      a.click();
    } else if (format === 'json') {
      const json = JSON.stringify(generatedData, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `synthetic_${activeTab}_data.json`;
      a.click();
    }
  };

  const getChartData = () => {
    if (!generatedData) return [];

    if (activeTab === 'radiosonde') {
      return generatedData.slice(0, 50); // Limit for performance
    } else {
      return generatedData.filter((_, i) => i % 4 === 0); // Sample every 2 minutes
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 mb-6 border border-white/20">
          <h1 className="text-4xl font-bold text-white mb-2">
            Synthetic Atmospheric & Aviation Dataset Generator
          </h1>
          <p className="text-blue-200">Generate realistic atmospheric and aviation data for research and simulation</p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-4 mb-6 flex-wrap">
          <button
            onClick={() => setActiveTab('radiosonde')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${activeTab === 'radiosonde'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
          >
            <Cloud size={20} />
            Radiosonde
          </button>
          <button
            onClick={() => setActiveTab('aviation')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${activeTab === 'aviation'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
          >
            <Plane size={20} />
            Aviation
          </button>
          <button
            onClick={() => setActiveTab('image')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${activeTab === 'image'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
          >
            <Image size={20} />
            Imagery
          </button>
          <button
            onClick={() => setActiveTab('evaluation')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${activeTab === 'evaluation'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
          >
            <BarChart2 size={20} />
            Evaluation
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${activeTab === 'training'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-white/10 text-blue-200 hover:bg-white/20'
              }`}
          >
            <Activity size={20} />
            Training
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Parameters / Controls Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Settings size={20} />
                {activeTab === 'image' ? 'Image Settings' :
                  activeTab === 'evaluation' ? 'Evaluation Controls' :
                    activeTab === 'training' ? 'Model Training Demo' : 'Parameters'}
              </h2>

              {/* Image Generation Controls */}
              {activeTab === 'image' && (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-blue-200 block mb-2">Image Type</label>
                    <select
                      value={imageType}
                      onChange={(e) => setImageType(e.target.value)}
                      className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                    >
                      <option value="satellite">Satellite Imagery</option>
                      <option value="radar">Radar Reflectivity</option>
                      <option value="pressure">Pressure Map</option>
                    </select>
                  </div>
                  <button
                    onClick={generateImage}
                    disabled={loading}
                    className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? 'Generating...' : (
                      <>
                        <Image size={20} />
                        Generate Image
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Evaluation Controls */}
              {activeTab === 'evaluation' && (
                <div className="space-y-4">
                  <p className="text-blue-200 text-sm">
                    Run a comprehensive evaluation of the generator against physics-based constraints and historical data distributions.
                  </p>
                  <button
                    onClick={evaluateModel}
                    disabled={loading}
                    className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? 'Evaluating...' : (
                      <>
                        <BarChart2 size={20} />
                        Run Evaluation
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Training Controls */}
              {activeTab === 'training' && (
                <div className="space-y-4">
                  <p className="text-blue-200 text-sm">
                    Train two separate downstream models (Random Forest) on Real vs Generated data and compare their performance on a held-out test set.
                  </p>
                  <div className="bg-white/5 p-3 rounded-lg text-xs text-blue-300">
                    <strong>Task:</strong> Predict Temperature from Altitude (Radiosonde)
                  </div>
                  <button
                    onClick={evaluateTraining}
                    disabled={loading}
                    className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? 'Training & Evaluating...' : (
                      <>
                        <Activity size={20} />
                        Run Training Comparison
                      </>
                    )}
                  </button>
                </div>
              )}




              {/* Standard Data Generation Controls */}
              {(activeTab === 'radiosonde' || activeTab === 'aviation') && (
                <>
                  <div className="mb-6">
                    <label className="text-sm text-blue-200 block mb-2">{activeTab === 'radiosonde' ? 'Weather Preset' : 'Flight Preset'}</label>
                    <select
                      className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                      onChange={(e) => e.target.value && applyPreset(activeTab === 'radiosonde' ? 'weather' : 'flight', e.target.value)}
                    >
                      <option value="">Select preset...</option>
                      {activeTab === 'radiosonde' ? (
                        <>
                          <option value="clear">Clear Sky</option>
                          <option value="stormy">Stormy</option>
                          <option value="winter">Winter</option>
                          <option value="summer">Summer</option>
                        </>
                      ) : (
                        <>
                          <option value="short_haul">Short Haul</option>
                          <option value="medium_haul">Medium Haul</option>
                          <option value="long_haul">Long Haul</option>
                        </>
                      )}
                    </select>
                  </div>

                  <div className="space-y-4">
                    {activeTab === 'radiosonde' ? (
                      <>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Min Altitude (m)</label>
                          <input
                            type="number"
                            value={parameters.min_altitude}
                            onChange={(e) => handleParameterChange('min_altitude', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Max Altitude (m)</label>
                          <input
                            type="number"
                            value={parameters.max_altitude}
                            onChange={(e) => handleParameterChange('max_altitude', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Data Points</label>
                          <input
                            type="number"
                            value={parameters.num_points}
                            onChange={(e) => handleParameterChange('num_points', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Surface Temp (°C)</label>
                          <input
                            type="number"
                            value={parameters.surface_temp}
                            onChange={(e) => handleParameterChange('surface_temp', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Surface Pressure (hPa)</label>
                          <input
                            type="number"
                            value={parameters.surface_pressure}
                            onChange={(e) => handleParameterChange('surface_pressure', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Surface Humidity (%)</label>
                          <input
                            type="number"
                            value={parameters.surface_humidity}
                            onChange={(e) => handleParameterChange('surface_humidity', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Duration (min)</label>
                          <input
                            type="number"
                            value={parameters.duration_minutes}
                            onChange={(e) => handleParameterChange('duration_minutes', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Cruise Altitude (m)</label>
                          <input
                            type="number"
                            value={parameters.cruise_altitude}
                            onChange={(e) => handleParameterChange('cruise_altitude', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                        <div>
                          <label className="text-sm text-blue-200 block mb-1">Cruise Speed (m/s)</label>
                          <input
                            type="number"
                            value={parameters.cruise_speed}
                            onChange={(e) => handleParameterChange('cruise_speed', e.target.value)}
                            className="w-full p-2 rounded-lg bg-white/10 text-white border border-white/20"
                          />
                        </div>
                      </>
                    )}
                  </div>

                  <button
                    onClick={generateData}
                    disabled={loading}
                    className="w-full mt-6 px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {loading ? 'Generating...' : (
                      <>
                        <Database size={20} />
                        Generate Data
                      </>
                    )}
                  </button>

                  {generatedData && (
                    <div className="flex gap-2 mt-4">
                      <button onClick={() => downloadData('csv')} className="flex-1 px-3 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-all flex items-center justify-center gap-2">
                        <Download size={16} /> CSV
                      </button>
                      <button onClick={() => downloadData('json')} className="flex-1 px-3 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-all flex items-center justify-center gap-2">
                        <Download size={16} /> JSON
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Validation Panel (Radiosonde only) */}
            {validationResult && (activeTab === 'radiosonde' || activeTab === 'aviation') && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 mt-6">
                <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                  {validationResult.valid ? <><CheckCircle className="text-green-400" size={20} /> Validation Passed</> : <><AlertCircle className="text-red-400" size={20} /> Validation Issues</>}
                </h3>
                {validationResult.issues.map((issue: any, i: number) => <div key={i} className="text-sm text-red-300">• {issue}</div>)}
                {validationResult.warnings.map((warn: any, i: number) => <div key={i} className="text-sm text-yellow-300">⚠ {warn}</div>)}
              </div>
            )}
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-2 space-y-6">



            {/* Training Comparison View */}
            {activeTab === 'training' && trainingMetrics && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Real Model Metrics */}
                  <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                    <h3 className="text-lg font-bold text-blue-300 mb-2">Model A (Trained on Real Data)</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">R² Score</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.real_model.r2.toFixed(4)}</div>
                      </div>
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">MAE</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.real_model.mae.toFixed(4)}</div>
                      </div>
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">RMSE</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.real_model.rmse.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>

                  {/* Synthetic Model Metrics */}
                  <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                    <h3 className="text-lg font-bold text-purple-300 mb-2">Model B (Trained on Generated Data)</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">R² Score</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.synthetic_model.r2.toFixed(4)}</div>
                      </div>
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">MAE</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.synthetic_model.mae.toFixed(4)}</div>
                      </div>
                      <div className="bg-white/5 p-3 rounded-lg">
                        <div className="text-xs text-blue-200">RMSE</div>
                        <div className="text-xl font-bold text-white">{trainingMetrics.metrics.synthetic_model.rmse.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Training Visualization */}
                <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                  <h3 className="text-xl font-bold text-white mb-4">Downstream Model predictions on Test Set</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={trainingMetrics.visualization}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="altitude" stroke="rgba(255,255,255,0.5)" label={{ value: 'Altitude (m)', style: { fill: 'white' } }} />
                      <YAxis stroke="rgba(255,255,255,0.5)" label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', style: { fill: 'white' } }} />
                      <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }} />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#10b981" name="Actual Values" dot={true} strokeWidth={0} />
                      <Line type="monotone" dataKey="pred_real_model" stroke="#3b82f6" name="Real Model Pred" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="pred_syn_model" stroke="#8b5cf6" name="Synthetic Model Pred" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Evaluation View */}
            {activeTab === 'evaluation' && evaluationMetrics && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(evaluationMetrics.metrics.r2_score).map(([key, val]) => (
                    <div key={key} className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
                      <div className="text-xs text-blue-300 uppercase">{key} R²</div>
                      <div className="text-2xl font-bold text-white">{(val as number).toFixed(3)}</div>
                    </div>
                  ))}
                </div>

                <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                  <h3 className="text-xl font-bold text-white mb-4">Distribution Comparison (Temperature)</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={evaluationMetrics.distributions.bins.slice(0, -1).map((bin: any, i: number) => ({
                      bin: bin.toFixed(1),
                      Real: evaluationMetrics.distributions.real_counts[i],
                      Generated: evaluationMetrics.distributions.gan_counts[i]
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="bin" stroke="rgba(255,255,255,0.5)" />
                      <YAxis stroke="rgba(255,255,255,0.5)" />
                      <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }} labelStyle={{ color: 'white' }} />
                      <Legend />
                      <Bar dataKey="Real" fill="#3b82f6" name="Real Data" />
                      <Bar dataKey="Generated" fill="#10b981" name="Generated Data" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Image View */}
            {activeTab === 'image' && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 flex flex-col items-center justify-center min-h-[400px]">
                {generatedImage ? (
                  <div className="w-full">
                    <img src={generatedImage.image} alt="Generated" className="w-full rounded-lg shadow-2xl border border-white/10" />
                    <div className="mt-4 flex justify-between items-center text-sm text-blue-200">
                      <span>Type: {generatedImage.metadata.type}</span>
                      <span>Generated at: {new Date(generatedImage.metadata.generated_at).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center opacity-50">
                    <Image size={64} className="mx-auto mb-4 text-blue-300" />
                    <p className="text-xl text-blue-200">Select options and generate image</p>
                  </div>
                )}
              </div>
            )}

            {/* Standard Visualization View (Radiosonde/Aviation) */}
            {(activeTab === 'radiosonde' || activeTab === 'aviation') && (
              <>
                <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                  <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <Activity size={20} />
                    Data Visualization
                  </h2>
                  {generatedData ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getChartData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                          dataKey={activeTab === 'radiosonde' ? 'altitude_m' : 'time_min'}
                          stroke="rgba(255,255,255,0.5)"
                          label={{ value: activeTab === 'radiosonde' ? 'Altitude (m)' : 'Time (min)', style: { fill: 'rgba(255,255,255,0.5)' } }}
                        />
                        <YAxis stroke="rgba(255,255,255,0.5)" />
                        <Tooltip
                          contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                          labelStyle={{ color: 'white' }}
                        />
                        <Legend />
                        {activeTab === 'radiosonde' ? (
                          <>
                            <Line type="monotone" dataKey="temperature_c" stroke="#ef4444" name="Temperature (°C)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="wind_speed_mps" stroke="#22d3ee" name="Wind Speed (m/s)" strokeWidth={2} dot={false} />
                          </>
                        ) : (
                          <>
                            <Line type="monotone" dataKey="altitude_m" stroke="#10b981" name="Altitude (m)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="airspeed_mps" stroke="#f59e0b" name="Airspeed (m/s)" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="thrust_percent" stroke="#8b5cf6" name="Thrust (%)" strokeWidth={2} dot={false} />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center border-2 border-dashed border-white/20 rounded-lg">
                      <div className="text-center">
                        <Activity size={48} className="mx-auto mb-3 text-blue-300 opacity-50" />
                        <p className="text-blue-200 text-lg">No data to visualize</p>
                        <p className="text-blue-300 text-sm mt-1">Generate data to see charts here</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Statistics */}
                <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                  <h2 className="text-xl font-bold text-white mb-4">Dataset Statistics</h2>
                  {statistics ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {Object.entries(statistics).slice(0, 6).map(([key, stats]) => (
                        <div key={key} className="bg-white/5 rounded-lg p-3">
                          <div className="text-sm text-blue-200 mb-1">{key.replace(/_/g, ' ')}</div>
                          <div className="text-white font-semibold">
                            {(stats as any).mean.toFixed(2)}
                          </div>
                          <div className="text-xs text-blue-300">
                            Range: {(stats as any).min.toFixed(1)} - {(stats as any).max.toFixed(1)}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {['Altitude', 'Temperature', 'Pressure', 'Humidity', 'Wind Speed', 'Wind Direction'].map((label) => (
                        <div key={label} className="bg-white/5 rounded-lg p-3 border-2 border-dashed border-white/10">
                          <div className="text-sm text-blue-200 mb-1">{label}</div>
                          <div className="text-white font-semibold text-2xl">--</div>
                          <div className="text-xs text-blue-300">No data</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Data Preview */}
                <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                  <h2 className="text-xl font-bold text-white mb-4">Data Preview (First 10 Records)</h2>
                  {generatedData ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-white/20">
                            {Object.keys(generatedData[0]).filter(k => k !== 'index').map(key => (
                              <th key={key} className="text-left p-2 text-blue-200">
                                {key.replace(/_/g, ' ')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {generatedData.slice(0, 10).map((row: any, i: number) => (
                            <tr key={i} className="border-b border-white/10">
                              {Object.entries(row).filter(([key]) => key !== 'index').map(([key, value]) => (
                                <td key={key} className="p-2 text-white">
                                  {typeof value === 'number' ? value.toFixed(2) : (value as any)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="border-2 border-dashed border-white/20 rounded-lg p-8">
                      <div className="text-center">
                        <Database size={48} className="mx-auto mb-3 text-blue-300 opacity-50" />
                        <p className="text-blue-200 text-lg">No data available</p>
                        <p className="text-blue-300 text-sm mt-1">Click "Generate Data" to create synthetic dataset</p>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
