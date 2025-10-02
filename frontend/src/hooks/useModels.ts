/**
 * React hooks for model-related API calls
 */

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { modelsApi, Model } from '../lib/api';
import { toast } from 'sonner';

// Hook for getting all models
export const useModels = (params?: {
  skip?: number;
  limit?: number;
  model_type?: string;
  active_only?: boolean;
}) => {
  return useQuery({
    queryKey: ['models', params],
    queryFn: () => modelsApi.getModels(params),
    onError: (error: any) => {
      toast.error(`Failed to get models: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting a specific model
export const useModel = (modelId: number) => {
  return useQuery({
    queryKey: ['models', modelId],
    queryFn: () => modelsApi.getModel(modelId),
    enabled: !!modelId,
    onError: (error: any) => {
      toast.error(`Failed to get model: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting model metrics
export const useModelMetrics = (modelId: number) => {
  return useQuery({
    queryKey: ['models', modelId, 'metrics'],
    queryFn: () => modelsApi.getModelMetrics(modelId),
    enabled: !!modelId,
    onError: (error: any) => {
      toast.error(`Failed to get model metrics: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting model training history
export const useModelHistory = (modelId: number) => {
  return useQuery({
    queryKey: ['models', modelId, 'history'],
    queryFn: () => modelsApi.getModelHistory(modelId),
    enabled: !!modelId,
    onError: (error: any) => {
      toast.error(`Failed to get model history: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for activating a model
export const useActivateModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: number) => modelsApi.activateModel(modelId),
    onSuccess: (data, modelId) => {
      toast.success(data.message);
      queryClient.invalidateQueries(['models']);
      queryClient.invalidateQueries(['models', modelId]);
    },
    onError: (error: any) => {
      toast.error(`Failed to activate model: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for deleting a model
export const useDeleteModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (modelId: number) => modelsApi.deleteModel(modelId),
    onSuccess: (data, modelId) => {
      toast.success(data.message);
      queryClient.invalidateQueries(['models']);
      queryClient.removeQueries(['models', modelId]);
    },
    onError: (error: any) => {
      toast.error(`Failed to delete model: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting supported model types
export const useSupportedModelTypes = () => {
  return useQuery({
    queryKey: ['models', 'types'],
    queryFn: () => modelsApi.getSupportedTypes(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    onError: (error: any) => {
      toast.error(`Failed to get model types: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for comparing models
export const useCompareModels = () => {
  return useMutation({
    mutationFn: (modelIds: number[]) => modelsApi.compareModels(modelIds),
    onError: (error: any) => {
      toast.error(`Failed to compare models: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting model leaderboard
export const useModelLeaderboard = (metric: string = 'accuracy', limit: number = 10) => {
  return useQuery({
    queryKey: ['models', 'leaderboard', metric, limit],
    queryFn: async () => {
      // This would be implemented in the backend
      const response = await fetch(`/api/v1/models/performance/leaderboard?metric=${metric}&limit=${limit}`);
      return response.json();
    },
    staleTime: 2 * 60 * 1000, // 2 minutes
    onError: (error: any) => {
      toast.error(`Failed to get leaderboard: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Custom hook for model management
export const useModelManager = () => {
  const queryClient = useQueryClient();
  const activateModelMutation = useActivateModel();
  const deleteModelMutation = useDeleteModel();

  const activateModel = async (modelId: number) => {
    try {
      await activateModelMutation.mutateAsync(modelId);
    } catch (error) {
      throw error;
    }
  };

  const deleteModel = async (modelId: number) => {
    try {
      await deleteModelMutation.mutateAsync(modelId);
    } catch (error) {
      throw error;
    }
  };

  const refreshModels = () => {
    queryClient.invalidateQueries(['models']);
  };

  return {
    activateModel,
    deleteModel,
    refreshModels,
    isActivating: activateModelMutation.isLoading,
    isDeleting: deleteModelMutation.isLoading,
  };
};
