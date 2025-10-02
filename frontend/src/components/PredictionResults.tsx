import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Download, Eye, EyeOff } from 'lucide-react';
import { MetricsDisplay } from './MetricsDisplay';
import { VisualizationDisplay } from './VisualizationDisplay';

interface PredictionResultsProps {
  results: {
    predictions: number[];
    labels: string[];
    confidence: number[];
    // Batch prediction specific fields
    total_samples?: number;
    confirmed_count?: number;
    false_positive_count?: number;
    average_confidence?: number;
    // Metrics (available when true labels are provided)
    metrics?: {
      accuracy: number | null;
      precision: number | null;
      recall: number | null;
      f1_score: number | null;
      specificity: number | null;
      roc_auc: number | null;
      average_precision: number | null;
      true_positives: number | null;
      true_negatives: number | null;
      false_positives: number | null;
      false_negatives: number | null;
    };
    visualizations?: {
      roc_curve?: string;
      pr_curve?: string;
      confusion_matrix?: string;
      shap_importance?: string;
    };
  };
  onExportResults?: () => void;
  isBatch?: boolean;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({ 
  results, 
  onExportResults,
  isBatch = false
}) => {
  const [showDetails, setShowDetails] = React.useState(false);
  const [showVisualizations, setShowVisualizations] = React.useState(true);

  // Use batch-specific fields if available, otherwise calculate from arrays
  const confirmedCount = results.confirmed_count ?? results.labels.filter(label => label === 'CONFIRMED').length;
  const falsePositiveCount = results.false_positive_count ?? results.labels.filter(label => label === 'FALSE POSITIVE').length;
  const avgConfidence = results.average_confidence ?? (results.confidence.reduce((a, b) => a + b, 0) / results.confidence.length);
  const totalSamples = results.total_samples ?? results.predictions.length;

  const exportResults = () => {
    const csvContent = [
      ['Index', 'Prediction', 'Label', 'Confidence'],
      ...results.predictions.map((pred, index) => [
        index + 1,
        pred.toFixed(4),
        results.labels[index],
        results.confidence[index].toFixed(4)
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_results.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-semibold">
              {isBatch ? 'Batch Prediction Results' : 'Prediction Results'}
            </CardTitle>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDetails(!showDetails)}
              >
                {showDetails ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                {showDetails ? 'Hide Details' : 'Show Details'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={exportResults}
              >
                <Download className="h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{totalSamples}</div>
              <div className="text-sm text-blue-700">Total Predictions</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">{confirmedCount}</div>
              <div className="text-sm text-green-700">Confirmed Exoplanets</div>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="text-2xl font-bold text-red-600">{falsePositiveCount}</div>
              <div className="text-sm text-red-700">False Positives</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">{(avgConfidence * 100).toFixed(1)}%</div>
              <div className="text-sm text-purple-700">Avg Confidence</div>
            </div>
          </div>

          {showDetails && (
            <div className="space-y-4">
              <div className="max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Index</th>
                      <th className="text-left p-2">Prediction</th>
                      <th className="text-left p-2">Label</th>
                      <th className="text-left p-2">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.predictions.map((pred, index) => (
                      <tr key={index} className="border-b hover:bg-gray-50">
                        <td className="p-2">{index + 1}</td>
                        <td className="p-2">{pred.toFixed(4)}</td>
                        <td className="p-2">
                          <Badge 
                            variant={results.labels[index] === 'CONFIRMED' ? 'default' : 'secondary'}
                          >
                            {results.labels[index]}
                          </Badge>
                        </td>
                        <td className="p-2">{(results.confidence[index] * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Metrics Display */}
      {results.metrics ? (
        <MetricsDisplay metrics={results.metrics} />
      ) : isBatch ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Model Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              <p>Performance metrics are not available for this batch prediction.</p>
              <p className="text-sm mt-2">
                To see detailed metrics and confusion matrix, provide a true labels column in your CSV file.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {/* Visualizations */}
      {results.visualizations ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Model Analysis</h3>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowVisualizations(!showVisualizations)}
            >
              {showVisualizations ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {showVisualizations ? 'Hide Visualizations' : 'Show Visualizations'}
            </Button>
          </div>
          {showVisualizations && (
            <VisualizationDisplay visualizations={results.visualizations} />
          )}
        </div>
      ) : isBatch ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Model Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              <p>Visualizations are not available for this batch prediction.</p>
              <p className="text-sm mt-2">
                To see model analysis charts, provide a true labels column in your CSV file.
              </p>
            </div>
          </CardContent>
        </Card>
      ) : null}
      
      {/* Debug: Show visualization data when available */}
      {process.env.NODE_ENV === 'development' && results.visualizations && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Debug: Visualization Data</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-32">
              {JSON.stringify(results.visualizations, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
