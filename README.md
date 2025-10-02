# CosmicX – AI-Powered Exoplanet Hunter

## 🚀 Short Description

CosmicX is an advanced AI-powered platform that leverages machine learning to discover and analyze exoplanets using data from NASA's space missions. Our platform combines state-of-the-art deep learning models with intuitive visualization tools to help researchers and astronomers identify potential exoplanets from stellar light curves.

## 🌟 Introduction

In the vast expanse of space, exoplanets hold the key to understanding our universe and the potential for life beyond Earth. CosmicX revolutionizes exoplanet detection by employing cutting-edge artificial intelligence algorithms to analyze light curves from NASA's Kepler, K2, and TESS missions. Our platform transforms complex astronomical data into actionable insights, making exoplanet discovery more accessible and efficient than ever before.

## 👥 Team: Lost In Orbit

- **Team Leader & Researcher**: Abdus Samad
- **UI/UX Designer**: Asikhur Rahman  
- **Video Editor**: Asmani Akter
- **Developer**: Zehad Khan
- **Architect**: Aman Ullah

## ✨ Features

### 🔬 Machine Learning Models
- **XGBoost**: Gradient boosting for tabular feature analysis
- **1D CNN**: Convolutional neural networks for time series pattern recognition
- **1D ResNet**: Deep residual networks for complex feature extraction
- **Transformer**: Attention-based models for sequence modeling

### 📊 Data Management
- **Multi-Mission Support**: Kepler, K2, and TESS mission data integration
- **Dataset Upload**: CSV file upload and validation
- **Real-time Processing**: Live data processing and analysis
- **Data Visualization**: Interactive charts and graphs

### 🎯 Prediction Capabilities
- **Single Predictions**: Individual exoplanet candidate analysis
- **Batch Predictions**: Bulk processing of multiple candidates
- **Confidence Scoring**: AI-powered confidence metrics
- **Model Comparison**: Side-by-side model performance analysis

### 📈 Training & Analysis
- **Model Training**: Configure and train custom ML models
- **Progress Tracking**: Real-time training progress monitoring
- **Performance Metrics**: Comprehensive model evaluation
- **Model Persistence**: Save and load trained models

### 🎨 User Interface
- **Responsive Design**: Modern, intuitive web interface
- **Real-time Updates**: Live progress tracking and notifications
- **Interactive Visualizations**: Dynamic charts and data displays
- **Dark/Light Theme**: Customizable user experience

## 🛠️ Tools & Technologies

### Backend Technologies
- **FastAPI**: High-performance Python web framework
- **TensorFlow 2.16.1**: Deep learning framework
- **XGBoost**: Gradient boosting library
- **Scikit-learn**: Machine learning toolkit
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Frontend Technologies
- **React 18**: Modern JavaScript UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Radix UI**: Accessible component primitives
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Server state management
- **Axios**: HTTP client library
- **Recharts**: Chart library
- **Lucide React**: Icon library

### Development Tools
- **Python 3.12**: Programming language
- **Node.js**: JavaScript runtime
- **Git**: Version control
- **VS Code**: Development environment

## 🚀 Project Setup & Installation

### Prerequisites
- Python 3.8+ (Python 3.12 recommended)
- Node.js 18+
- Git

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Exoplanet AI Web App Design"
   ```

2. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend server**:
   ```bash
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Create environment file**:
   ```bash
   cp env.example .env
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```
   
   The application will be available at `http://localhost:5173`

## 📖 API Documentation

Once the backend is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

### Key API Endpoints

#### Dataset Management
- `POST /api/datasets/upload` - Upload dataset for a specific mission
- `GET /api/datasets` - List all uploaded datasets

#### Model Training
- `POST /api/training/start` - Start training models
- `GET /api/training/status/{training_id}` - Get training status
- `GET /api/training/history` - Get training history

#### Predictions
- `POST /api/prediction/predict` - Make predictions
- `POST /api/prediction/batch` - Batch predictions
- `GET /api/models/available` - List available models

## 🎯 How to Use

1. **Upload Data**: Upload CSV files containing exoplanet data from Kepler, K2, or TESS missions
2. **Configure Training**: Select models and parameters for training
3. **Train Models**: Start training and monitor progress in real-time
4. **Make Predictions**: Use trained models to predict exoplanet candidates
5. **Analyze Results**: View predictions, confidence scores, and visualizations

## 📊 Dataset Format

The platform supports CSV files with mission-specific columns:

### Kepler Mission
- Required: `koi_disposition`, `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`
- Optional: `koi_score`, `koi_fpflag_*`, `koi_steff`, `koi_slogg`, `koi_srad`

### K2 Mission  
- Required: `disposition`, `pl_orbper`, `pl_trandur`, `pl_trandep`, `pl_rade`
- Optional: Various stellar and planetary parameters

### TESS Mission
- Required: `tfopwg_disp`, `toi_period`, `toi_duration`, `toi_depth`
- Optional: Various TOI and stellar parameters

## 🖼️ Screenshots

*Screenshots will be added manually*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA for providing the Kepler, K2, and TESS mission data
- The open-source community for the amazing tools and libraries
- Our team members for their dedication and hard work

## 📞 Contact

For questions or support, please contact:
- **Team Leader**: Abdus Samad
- **Email**: [contact-email]
- **Project Repository**: [repository-url]

---

**Lost In Orbit Team** - Exploring the cosmos, one exoplanet at a time 🌌
