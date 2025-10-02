/**
 * React hooks for training-related API calls
 */

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { trainingApi, TrainingConfig, TrainingRun } from '../lib/api';
import { toast } from 'sonner';

// Hook for starting training
export const useStartTraining = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (config: TrainingConfig) => trainingApi.startTraining(config),
    onSuccess: (data) => {
      toast.success(`Training started with run ID: ${data.run_id}`);
      queryClient.invalidateQueries(['training', 'jobs']);
      queryClient.invalidateQueries(['training', 'runs']);
    },
    onError: (error: any) => {
      toast.error(`Failed to start training: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting training status
export const useTrainingStatus = (runId: string | null, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['training', 'status', runId],
    queryFn: () => trainingApi.getTrainingStatus(runId!),
    enabled: enabled && !!runId,
    refetchInterval: (data) => {
      // Stop polling if training is completed or failed
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false;
      }
      return 2000; // Poll every 2 seconds for active training
    },
    onError: (error: any) => {
      toast.error(`Failed to get training status: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for stopping training
export const useStopTraining = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (runId: string) => trainingApi.stopTraining(runId),
    onSuccess: (data, runId) => {
      toast.success(data.message);
      queryClient.invalidateQueries(['training', 'status', runId]);
      queryClient.invalidateQueries(['training', 'jobs']);
    },
    onError: (error: any) => {
      toast.error(`Failed to stop training: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting active training jobs
export const useActiveTrainingJobs = () => {
  return useQuery({
    queryKey: ['training', 'jobs'],
    queryFn: () => trainingApi.getActiveJobs(),
    refetchInterval: 5000, // Poll every 5 seconds
    onError: (error: any) => {
      toast.error(`Failed to get active jobs: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for getting training runs history
export const useTrainingRuns = (params?: {
  skip?: number;
  limit?: number;
  status?: string;
}) => {
  return useQuery({
    queryKey: ['training', 'runs', params],
    queryFn: () => trainingApi.getTrainingRuns(params),
    onError: (error: any) => {
      toast.error(`Failed to get training runs: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Hook for validating training config
export const useValidateTrainingConfig = () => {
  return useMutation({
    mutationFn: (config: TrainingConfig) => trainingApi.validateConfig(config),
    onError: (error: any) => {
      toast.error(`Invalid configuration: ${error.response?.data?.detail || error.message}`);
    },
  });
};

// Custom hook for managing training state
export const useTrainingManager = () => {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  const startTrainingMutation = useStartTraining();
  const stopTrainingMutation = useStopTraining();
  const { data: trainingStatus, isLoading: statusLoading } = useTrainingStatus(
    currentRunId,
    isTraining
  );

  // Update training state based on status
  useEffect(() => {
    if (trainingStatus) {
      const isActive = trainingStatus.status === 'running' || trainingStatus.status === 'pending';
      setIsTraining(isActive);

      // Clear run ID if training is completed or failed
      if (trainingStatus.status === 'completed' || trainingStatus.status === 'failed') {
        setTimeout(() => {
          setCurrentRunId(null);
          setIsTraining(false);
        }, 5000); // Clear after 5 seconds
      }
    }
  }, [trainingStatus]);

  const startTraining = useCallback(
    async (config: TrainingConfig) => {
      try {
        const result = await startTrainingMutation.mutateAsync(config);
        setCurrentRunId(result.run_id);
        setIsTraining(true);
        return result;
      } catch (error) {
        throw error;
      }
    },
    [startTrainingMutation]
  );

  const stopTraining = useCallback(
    async (runId?: string) => {
      const targetRunId = runId || currentRunId;
      if (!targetRunId) return;

      try {
        await stopTrainingMutation.mutateAsync(targetRunId);
        setIsTraining(false);
        setCurrentRunId(null);
      } catch (error) {
        throw error;
      }
    },
    [stopTrainingMutation, currentRunId]
  );

  return {
    currentRunId,
    isTraining,
    trainingStatus,
    statusLoading,
    startTraining,
    stopTraining,
    isStarting: startTrainingMutation.isLoading,
    isStopping: stopTrainingMutation.isLoading,
  };
};
