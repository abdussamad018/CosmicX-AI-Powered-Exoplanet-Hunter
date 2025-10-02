import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Trash2, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  RefreshCw,
  Database,
  FileText,
  HardDrive
} from 'lucide-react';
import { resetApi } from '../lib/api';

interface DataSummary {
  trained_models: number;
  datasets: number;
  total_size_mb: number;
}

interface ResetResults {
  models_removed: number;
  datasets_removed: number;
  temp_files_removed: number;
  errors: string[];
}

export function Reset() {
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [resetResults, setResetResults] = useState<ResetResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);

  // Load data summary on component mount
  useEffect(() => {
    loadDataSummary();
  }, []);

  const loadDataSummary = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const summary = await resetApi.getDataSummary();
      setDataSummary(summary);
    } catch (err) {
      setError('Failed to load data summary');
      console.error('Error loading data summary:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      setIsResetting(true);
      setError(null);
      setResetResults(null);
      
      const results = await resetApi.resetAllData();
      setResetResults(results);
      setShowConfirmDialog(false);
      
      // Reload data summary after reset
      await loadDataSummary();
      
    } catch (err) {
      setError('Failed to reset data');
      console.error('Error resetting data:', err);
    } finally {
      setIsResetting(false);
    }
  };

  const formatFileSize = (sizeInMB: number) => {
    if (sizeInMB < 1) {
      return `${(sizeInMB * 1024).toFixed(1)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Reset Data</h1>
        <p className="text-muted-foreground">
          Clear all training models, datasets, and temporary files for a fresh start
        </p>
      </div>

      <div className="space-y-6">
        {/* Current Data Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Database className="w-5 h-5 mr-2" />
              Current Data Summary
            </CardTitle>
            <CardDescription>
              Overview of data that will be removed during reset
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="w-6 h-6 animate-spin mr-2" />
                <span>Loading data summary...</span>
              </div>
            ) : dataSummary ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="text-2xl font-bold text-blue-600">{dataSummary.trained_models}</div>
                  <div className="text-sm text-blue-700">Trained Models</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="text-2xl font-bold text-green-600">{dataSummary.datasets}</div>
                  <div className="text-sm text-green-700">Datasets</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <div className="text-2xl font-bold text-purple-600">{formatFileSize(dataSummary.total_size_mb)}</div>
                  <div className="text-sm text-purple-700">Total Size</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <XCircle className="w-12 h-12 mx-auto mb-4" />
                <p>Failed to load data summary</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Reset Action */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Trash2 className="w-5 h-5 mr-2" />
              Reset All Data
            </CardTitle>
            <CardDescription>
              This action will permanently delete all training models, datasets, and temporary files
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!showConfirmDialog ? (
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Warning:</strong> This action cannot be undone. All trained models, 
                    uploaded datasets, and temporary files will be permanently deleted.
                  </AlertDescription>
                </Alert>
                
                <div className="flex gap-2">
                  <Button 
                    variant="destructive" 
                    onClick={() => setShowConfirmDialog(true)}
                    disabled={isLoading || !dataSummary}
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Reset All Data
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={loadDataSummary}
                    disabled={isLoading}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh Summary
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Final Confirmation:</strong> Are you sure you want to delete all data? 
                    This action cannot be undone.
                  </AlertDescription>
                </Alert>
                
                <div className="flex gap-2">
                  <Button 
                    variant="destructive" 
                    onClick={handleReset}
                    disabled={isResetting}
                  >
                    {isResetting ? (
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4 mr-2" />
                    )}
                    {isResetting ? 'Resetting...' : 'Yes, Delete All Data'}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => setShowConfirmDialog(false)}
                    disabled={isResetting}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Reset Results */}
        {resetResults && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
                Reset Completed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                    <div className="text-2xl font-bold text-red-600">{resetResults.models_removed}</div>
                    <div className="text-sm text-red-700">Models Removed</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <div className="text-2xl font-bold text-orange-600">{resetResults.datasets_removed}</div>
                    <div className="text-sm text-orange-700">Datasets Removed</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-2xl font-bold text-gray-600">{resetResults.temp_files_removed}</div>
                    <div className="text-sm text-gray-700">Temp Files Removed</div>
                  </div>
                </div>
                
                {resetResults.errors.length > 0 && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      <strong>Errors encountered:</strong>
                      <ul className="mt-2 list-disc list-inside">
                        {resetResults.errors.map((error, index) => (
                          <li key={index} className="text-sm">{error}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <Alert>
            <XCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
}
