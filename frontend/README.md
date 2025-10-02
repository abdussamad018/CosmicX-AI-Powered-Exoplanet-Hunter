# Exoplanet AI Frontend

A React-based frontend for the Exoplanet AI machine learning platform that enables users to train models, make predictions, and analyze exoplanet data.

## Features

- **Model Training**: Configure and train machine learning models for exoplanet detection
- **Real-time Progress**: Live updates during training with progress bars and metrics
- **Predictions**: Single and batch prediction capabilities with file upload support
- **Model Management**: View, compare, and manage trained models
- **Dataset Management**: Upload and manage exoplanet datasets
- **Interactive UI**: Modern, responsive interface with real-time feedback

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Radix UI** - Accessible component primitives
- **React Query** - Server state management
- **Axios** - HTTP client
- **Lucide React** - Icons
- **Sonner** - Toast notifications

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp env.example .env
```

3. Start development server:
```bash
npm run dev
```

The application will be available at http://localhost:5173

## API Integration

The frontend communicates with the FastAPI backend through:

- **Training API**: Model training, status tracking, and management
- **Prediction API**: Single and batch predictions with file upload
- **Models API**: Model listing, comparison, and metrics
- **Datasets API**: Dataset management and NASA data integration

### Key Components

- **ModelLab**: Training configuration and progress monitoring
- **Predict**: Single and batch prediction interface
- **API Hooks**: Custom React hooks for API communication
- **Real-time Updates**: Live training progress and status updates

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ui/             # Reusable UI components
│   │   ├── ModelLab.tsx    # Training interface
│   │   ├── Predict.tsx     # Prediction interface
│   │   └── ...
│   ├── hooks/              # Custom React hooks
│   │   ├── useTraining.ts  # Training API hooks
│   │   ├── useModels.ts    # Models API hooks
│   │   ├── usePredictions.ts # Prediction API hooks
│   │   └── useDatasets.ts  # Datasets API hooks
│   ├── lib/                # Utilities and configuration
│   │   ├── api.ts          # API client and types
│   │   └── react-query.ts  # React Query configuration
│   └── ...
├── package.json
└── README.md
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Environment Variables

- `VITE_API_URL` - Backend API URL (default: http://localhost:8000)

### API Configuration

The API client is configured in `src/lib/api.ts` with:
- Base URL configuration
- Request/response interceptors
- Error handling
- TypeScript types for all API endpoints

## Features Overview

### Model Training
- Select model architecture (CNN, ResNet, XGBoost, Transformer)
- Configure hyperparameters with interactive sliders
- Choose features from available dataset columns
- Real-time training progress with metrics
- Training history and comparison

### Predictions
- Single prediction with JSON feature input
- File upload for batch processing (CSV, FITS)
- Real-time prediction results with confidence scores
- Batch job tracking and progress monitoring

### Model Management
- View all trained models with performance metrics
- Compare model performance
- Activate/deactivate models
- Training history and model versions

### Dataset Management
- Upload custom datasets
- Download NASA datasets (Kepler, K2, TESS)
- Dataset statistics and preview
- Data validation and processing

## Integration with Backend

The frontend integrates seamlessly with the FastAPI backend:

1. **Authentication**: Ready for future authentication implementation
2. **Real-time Updates**: WebSocket support for live training progress
3. **File Upload**: Multipart form data for dataset and prediction files
4. **Error Handling**: Comprehensive error handling with user feedback
5. **Type Safety**: Full TypeScript integration with backend API types

## Deployment

### Production Build

```bash
npm run build
```

### Environment Configuration

Set production environment variables:
```env
VITE_API_URL=https://your-api-domain.com
```

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.