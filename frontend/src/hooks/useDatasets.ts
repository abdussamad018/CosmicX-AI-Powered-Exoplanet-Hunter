/**
 * React hooks for dataset-related API calls
 */

import { useQuery, useMutation, useQueryClient } from 'react-query';
import { datasetsApi, Dataset } from '../lib/api';
import { toast } from 'sonner';

// Hook for getting all datasets
export const useDatasets = (params?: {
  skip?: number;
  limit?: number;
  mission?: string;
  active_only?: boolean;
}) => {
  return useQuery({
    queryKey: ['datasets', params],
    queryFn: () => datasetsApi.getDatasets(params),
    onError: (error: any) => {
      toast.error(`Failed to get datasets: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting a specific dataset
export const useDataset = (datasetId: number) => {
  return useQuery({
    queryKey: ['datasets', datasetId],
    queryFn: () => datasetsApi.getDataset(datasetId),
    enabled: !!datasetId,
    onError: (error: any) => {
      toast.error(`Failed to get dataset: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for uploading a dataset
export const useUploadDataset = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ file, mission, description }: {
      file: File;
      mission?: string;
      description?: string;
    }) => datasetsApi.uploadDataset(file, mission, description),
    onSuccess: (data) => {
      toast.success(`Dataset "${data.name}" uploaded successfully`);
      queryClient.invalidateQueries(['datasets']);
    },
    onError: (error: any) => {
      toast.error(`Failed to upload dataset: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for downloading NASA dataset
export const useDownloadNASADataset = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (mission: string) => datasetsApi.downloadNASADataset(mission),
    onSuccess: (data, mission) => {
      toast.success(`NASA ${mission.toUpperCase()} dataset downloaded successfully`);
      queryClient.invalidateQueries(['datasets']);
    },
    onError: (error: any) => {
      toast.error(`Failed to download NASA dataset: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting dataset statistics
export const useDatasetStatistics = (datasetId: number) => {
  return useQuery({
    queryKey: ['datasets', datasetId, 'statistics'],
    queryFn: () => datasetsApi.getDatasetStatistics(datasetId),
    enabled: !!datasetId,
    onError: (error: any) => {
      toast.error(`Failed to get dataset statistics: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting dataset preview
export const useDatasetPreview = (datasetId: number, rows: number = 10) => {
  return useQuery({
    queryKey: ['datasets', datasetId, 'preview', rows],
    queryFn: () => datasetsApi.getDatasetPreview(datasetId, rows),
    enabled: !!datasetId,
    onError: (error: any) => {
      toast.error(`Failed to get dataset preview: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting available missions
export const useAvailableMissions = () => {
  return useQuery({
    queryKey: ['datasets', 'missions'],
    queryFn: () => datasetsApi.getAvailableMissions(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    onError: (error: any) => {
      toast.error(`Failed to get available missions: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Custom hook for dataset management
export const useDatasetManager = () => {
  const queryClient = useQueryClient();
  const uploadDatasetMutation = useUploadDataset();
  const downloadNASAMutation = useDownloadNASADataset();

  const uploadDataset = async (file: File, mission?: string, description?: string) => {
    try {
      const result = await uploadDatasetMutation.mutateAsync({ file, mission, description });
      return result;
    } catch (error) {
      throw error;
    }
  };

  const downloadNASADataset = async (mission: string) => {
    try {
      const result = await downloadNASAMutation.mutateAsync(mission);
      return result;
    } catch (error) {
      throw error;
    }
  };

  const refreshDatasets = () => {
    queryClient.invalidateQueries(['datasets']);
  };

  return {
    uploadDataset,
    downloadNASADataset,
    refreshDatasets,
    isUploading: uploadDatasetMutation.isLoading,
    isDownloading: downloadNASAMutation.isLoading,
  };
};
