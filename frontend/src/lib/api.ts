/**
 * API client configuration and base functions
 */

import axios, {AxiosInstance, AxiosResponse} from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
    baseURL: `${API_BASE_URL}/api`,
    timeout: 30000, // 30 seconds
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
    (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
    },
    (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
    (response: AxiosResponse) => {
        return response;
    },
    (error) => {
        console.error('[API] Response error:', error.response?.data || error.message);

        // Handle common error cases
        if (error.response?.status === 401) {
            // Handle unauthorized
            console.warn('[API] Unauthorized access');
        } else if (error.response?.status >= 500) {
            // Handle server errors
            console.error('[API] Server error:', error.response?.data);
        }

        return Promise.reject(error);
    }
);

// Generic API response type
export interface ApiResponse<T = any> {
    data: T;
    message?: string;
    status: string;
}

// API Error type
export interface ApiError {
    error: string;
    status_code: number;
    detail?: string;
}

// Training API types
export interface TrainingRequest {
    mission: string;
    dataset_filename: string;
    models: string[]; // ["xgboost", "cnn1d", "resnet1d", "transformer"]
    test_size: number;
}

export interface TrainingResponse {
    training_id: string;
    mission: string;
    models: string[];
    status: string;
    message: string;
}

export interface TrainingStatus {
    training_id: string;
    mission: string;
    models: string[];
    status: 'started' | 'loading_data' | 'training' | 'saving' | 'completed' | 'failed';
    progress: number;
    metrics?: Record<string, any>;
    error?: string;
    start_time?: string;
    end_time?: string;
}

export interface Model {
    id: number;
    name: string;
    model_type: string;
    version: string;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    roc_auc?: number;
    created_at: string;
    is_active: boolean;
    features_used: string[];
    hyperparameters: Record<string, any>;
}

export interface PredictionRequest {
  mission: string;
  dataset_filename: string;
  model_name: string;
  data: Array<Record<string, any>>;
  true_labels?: number[];
}

export interface PredictionResponse {
  predictions: number[];
  labels: string[];
  confidence: number[];
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    specificity: number;
    roc_auc: number;
    average_precision: number;
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  visualizations?: {
    roc_curve?: string;
    pr_curve?: string;
    confusion_matrix?: string;
    shap_importance?: string;
  };
}

export interface BatchPredictionRequest {
  mission: string;
  dataset_filename: string;
  model_name: string;
  file: File;
  true_labels_column?: string;
}

export interface BatchPredictionResponse {
  predictions: number[];
  labels: string[];
  confidence: number[];
  total_samples: number;
  confirmed_count: number;
  false_positive_count: number;
  average_confidence: number;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    specificity: number;
    roc_auc: number;
    average_precision: number;
    true_positives: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
  };
  visualizations?: {
    roc_curve?: string;
    pr_curve?: string;
    confusion_matrix?: string;
    shap_importance?: string;
  };
}

export interface Dataset {
    mission: string;
    filename: string;
    size: number;
    created: string;
    path: string;
}

// Training API functions
export const trainingApi = {
    // Start training
    startTraining: async (request: TrainingRequest): Promise<TrainingResponse> => {
        const response = await apiClient.post('/training/start', request);
        return response.data;
    },

    // Get training status
    getTrainingStatus: async (trainingId: string): Promise<TrainingStatus> => {
        const response = await apiClient.get(`/training/status/${trainingId}`);
        return response.data;
    },

    // Get training history
    getTrainingHistory: async (): Promise<TrainingStatus[]> => {
        const response = await apiClient.get('/training/history');
        return response.data.training_history;
    },
};

// Models API functions
export const modelsApi = {
    // Get available models
    getAvailableModels: async (): Promise<any[]> => {
        const response = await apiClient.get('/models/available');
        return response.data.models;
    },

    // Get model metrics
    getModelMetrics: async (modelId: string): Promise<Record<string, any>> => {
        const response = await apiClient.get(`/models/${modelId}/metrics`);
        return response.data.metrics;
    },
};

// Prediction API functions
export const predictionApi = {
    // Make prediction
    predict: async (request: PredictionRequest): Promise<PredictionResponse> => {
        const response = await apiClient.post('/prediction/predict', request);
        return response.data;
    },

    // Batch prediction
    predictBatch: async (request: BatchPredictionRequest): Promise<BatchPredictionResponse> => {
        const formData = new FormData();
        formData.append('mission', request.mission);
        formData.append('dataset_filename', request.dataset_filename);
        formData.append('model_name', request.model_name);
        formData.append('file', request.file);
        if (request.true_labels_column) {
            formData.append('true_labels_column', request.true_labels_column);
        }
        
        const response = await apiClient.post('/prediction/batch', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
};

// Datasets API functions
export const datasetsApi = {
    // Upload dataset
    uploadDataset: async (mission: string, file: File): Promise<{
        mission: string;
        filename: string;
        rows: number;
        columns: number;
        message: string;
    }> => {
        const formData = new FormData();
        formData.append('mission', mission);
        formData.append('file', file);

        const response = await apiClient.post('/datasets/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Get all datasets
    getDatasets: async (): Promise<{ datasets: Dataset[] }> => {
        const response = await apiClient.get('/datasets');
        return response.data;
    },
};

// Reset API functions
export const resetApi = {
    // Get data summary before reset
    getDataSummary: async (): Promise<{
        trained_models: number;
        datasets: number;
        total_size_mb: number;
    }> => {
        const response = await apiClient.get('/reset/summary');
        return response.data.data;
    },

    // Reset all data
    resetAllData: async (): Promise<{
        models_removed: number;
        datasets_removed: number;
        temp_files_removed: number;
        errors: string[];
    }> => {
        const response = await apiClient.post('/reset/all');
        return response.data.data;
    },
};

// Health check
export const healthApi = {
    checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
        const response = await apiClient.get('/health');
        return response.data;
    },
};

export default apiClient;
