/**
 * React hooks for prediction-related API calls
 */

import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { predictionApi, PredictionRequest, BatchPredictionRequest } from '../lib/api';
import { toast } from 'sonner';

// Hook for single prediction
export const useSinglePrediction = () => {
  return useMutation({
    mutationFn: (request: PredictionRequest) => predictionApi.predictSingle(request),
    onError: (error: any) => {
      toast.error(`Prediction failed: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for batch prediction
export const useBatchPrediction = () => {
  return useMutation({
    mutationFn: (request: BatchPredictionRequest) => predictionApi.predictBatch(request),
    onSuccess: (data) => {
      toast.success(`Batch prediction completed. Processed ${data.processed_samples} samples.`);
    },
    onError: (error: any) => {
      toast.error(`Batch prediction failed: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for file-based prediction
export const useFilePrediction = () => {
  return useMutation({
    mutationFn: ({ file, modelId, threshold }: {
      file: File;
      modelId: number;
      threshold: number;
    }) => predictionApi.predictFromFile(file, modelId, threshold),
    onSuccess: (data) => {
      toast.success(`File processed successfully. Found ${data.predictions.length} predictions.`);
    },
    onError: (error: any) => {
      toast.error(`File prediction failed: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting predictions history
export const usePredictions = (params?: {
  skip?: number;
  limit?: number;
  model_id?: number;
  status?: string;
}) => {
  return useQuery({
    queryKey: ['predictions', params],
    queryFn: () => predictionApi.getPredictions(params),
    onError: (error: any) => {
      toast.error(`Failed to get predictions: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting prediction statistics
export const usePredictionStatistics = (modelId?: number) => {
  return useQuery({
    queryKey: ['predictions', 'statistics', modelId],
    queryFn: () => predictionApi.getStatistics(modelId),
    onError: (error: any) => {
      toast.error(`Failed to get prediction statistics: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for validating features
export const useValidateFeatures = () => {
  return useMutation({
    mutationFn: ({ features, modelId }: { features: Record<string, number>; modelId: number }) =>
      predictionApi.validateFeatures(features, modelId),
    onError: (error: any) => {
      toast.error(`Feature validation failed: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Custom hook for prediction management
export const usePredictionManager = () => {
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  const singlePredictionMutation = useSinglePrediction();
  const batchPredictionMutation = useBatchPrediction();
  const filePredictionMutation = useFilePrediction();

  const predictSingle = useCallback(async (request: PredictionRequest) => {
    try {
      setIsPredicting(true);
      const result = await singlePredictionMutation.mutateAsync(request);
      setPredictionResult(result);
      return result;
    } catch (error) {
      setPredictionResult(null);
      throw error;
    } finally {
      setIsPredicting(false);
    }
  }, [singlePredictionMutation]);

  const predictBatch = useCallback(async (request: BatchPredictionRequest) => {
    try {
      setIsPredicting(true);
      const result = await batchPredictionMutation.mutateAsync(request);
      return result;
    } catch (error) {
      throw error;
    } finally {
      setIsPredicting(false);
    }
  }, [batchPredictionMutation]);

  const predictFromFile = useCallback(async (file: File, modelId: number, threshold: number = 0.5) => {
    try {
      setIsPredicting(true);
      const result = await filePredictionMutation.mutateAsync({ file, modelId, threshold });
      return result;
    } catch (error) {
      throw error;
    } finally {
      setIsPredicting(false);
    }
  }, [filePredictionMutation]);

  const clearResult = useCallback(() => {
    setPredictionResult(null);
  }, []);

  return {
    predictionResult,
    isPredicting,
    predictSingle,
    predictBatch,
    predictFromFile,
    clearResult,
    isSingleLoading: singlePredictionMutation.isLoading,
    isBatchLoading: batchPredictionMutation.isLoading,
    isFileLoading: filePredictionMutation.isLoading,
  };
};
