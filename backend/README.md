# Exoplanet AI Backend

A FastAPI-based backend service for training and deploying machine learning models for exoplanet detection using data from NASA missions (Kepler, K2, TESS).

## Features

- **Multi-Mission Support**: Handles datasets from Kepler, K2, and TESS missions
- **Multiple ML Models**: Supports XGBoost, 1D CNN, 1D ResNet, and Transformer models
- **Dataset Management**: Upload and process CSV datasets for different missions
- **Training Pipeline**: Train multiple models simultaneously with progress tracking
- **Prediction API**: Make predictions on new data using trained models
- **Model Persistence**: Save and load trained models with metadata

## Quick Start

### Prerequisites

- Python 3.8+ (Python 3.13 supported with special installation)
- pip or conda

### Installation

#### For Python 3.8-3.12 (Standard Installation)

1. Clone the repository and navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
# On Windows
start.bat

# On Linux/Mac
chmod +x start.sh
./start.sh

# Or manually
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### For Python 3.13 (Compatibility Installation)

Python 3.13 is very new and some packages may not have pre-built wheels yet. Use the special installation script:

**Windows:**
```bash
cd backend
install-py313.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x install-py313.sh
./install-py313.sh
```

**Manual Installation for Python 3.13:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6

# Install scientific libraries
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2

# Install ML libraries
pip install xgboost==2.1.2 tensorflow==2.16.1

# Install remaining dependencies
pip install matplotlib==3.9.2 seaborn==0.13.2 shap==0.45.0
pip install python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4
pip install sqlalchemy==2.0.36 alembic==1.13.2 psycopg2-binary==2.9.10
pip install pydantic==2.9.2 pydantic-settings==2.6.1
pip install python-dotenv==1.0.0 aiofiles==24.1.0
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## API Endpoints

### Dataset Management
- `POST /api/datasets/upload` - Upload a dataset for a specific mission
- `GET /api/datasets` - List all uploaded datasets

### Model Training
- `POST /api/training/start` - Start training models on a dataset
- `GET /api/training/status/{training_id}` - Get training status
- `GET /api/training/history` - Get training history

### Predictions
- `POST /api/prediction/predict` - Make predictions using a trained model
- `GET /api/models/available` - Get list of available trained models
- `GET /api/models/{model_id}/metrics` - Get metrics for a specific model

### Health Check
- `GET /api/health` - Check API health status

## Dataset Format

The backend expects CSV files with mission-specific columns:

### Kepler Mission
Required columns: `koi_disposition`, `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`
Optional columns: `koi_score`, `koi_fpflag_*`, `koi_steff`, `koi_slogg`, `koi_srad`

### K2 Mission
Required columns: `disposition` (or `koi_disposition`), `pl_orbper`, `pl_trandur`, `pl_trandep`, `pl_rade`
Optional columns: Various stellar and planetary parameters

### TESS Mission
Required columns: `tfopwg_disp` (or `koi_disposition`), `toi_period`, `toi_duration`, `toi_depth`
Optional columns: Various TOI and stellar parameters

## Model Training

### Supported Models

1. **XGBoost**: Gradient boosting for tabular features
2. **1D CNN**: Convolutional neural network for time series
3. **1D ResNet**: Residual network with skip connections
4. **Transformer**: Attention-based architecture

### Training Process

1. Upload a dataset using the `/api/datasets/upload` endpoint
2. Start training using `/api/training/start` with:
   - Mission type (kepler, k2, tess)
   - Dataset filename
   - List of models to train
   - Test split ratio
3. Monitor progress using `/api/training/status/{training_id}`
4. View results and metrics when training completes

## Making Predictions

### Single Prediction

```python
import requests

# Prepare prediction data
data = [{
    "koi_period": 365.25,
    "koi_duration": 6.2,
    "koi_depth": 84,
    "koi_prad": 1.0,
    "koi_steff": 5778,
    "koi_slogg": 4.44,
    "koi_srad": 1.0
}]

# Make prediction
response = requests.post("http://localhost:8000/api/prediction/predict", json={
    "mission": "kepler",
    "dataset_filename": "your_dataset.csv",
    "model_name": "xgboost",
    "data": data
})

result = response.json()
print(f"Prediction: {result['labels'][0]}")
print(f"Confidence: {result['confidence'][0]:.2%}")
```

## File Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── start.sh              # Linux/Mac startup script
├── start.bat             # Windows startup script
├── models/
│   ├── training.py       # Training manager
│   └── prediction.py     # Prediction manager
├── utils/
│   ├── file_handler.py   # File upload/management
│   └── data_processor.py # Data processing utilities
├── datasets/             # Uploaded datasets (created automatically)
│   ├── kepler/
│   ├── k2/
│   └── tess/
└── models/trained/       # Saved models (created automatically)
```

## Configuration

Environment variables can be set in `.env` file:

```env
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=100MB
LOG_LEVEL=INFO
```

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Test API endpoints
curl http://localhost:8000/api/health

# Test dataset upload
curl -X POST "http://localhost:8000/api/datasets/upload" \
  -F "mission=kepler" \
  -F "file=@your_dataset.csv"
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in the startup command
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **File upload errors**: Check file size limits and format
4. **Training failures**: Verify dataset format and required columns

### Logs

Check the console output for detailed error messages and training progress.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
