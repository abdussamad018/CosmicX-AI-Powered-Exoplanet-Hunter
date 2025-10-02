import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';

interface MetricsDisplayProps {
  metrics: {
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
}

export const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
  const formatPercentage = (value: number | null) => {
    if (value === null || value === undefined) return '0.00';
    return (value * 100).toFixed(2);
  };
  
  const formatDecimal = (value: number | null) => {
    if (value === null || value === undefined) return '0.000';
    return value.toFixed(3);
  };
  
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getScoreBadgeVariant = (score: number | null): "default" | "secondary" | "destructive" | "outline" => {
    if (score === null || score === undefined) return 'outline';
    if (score >= 0.8) return 'default';
    if (score >= 0.6) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl font-semibold">Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Primary Metrics */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Accuracy</span>
                <Badge variant={getScoreBadgeVariant(metrics.accuracy)}>
                  {formatPercentage(metrics.accuracy)}%
                </Badge>
              </div>
              <Progress 
                value={(metrics.accuracy || 0) * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Precision</span>
                <Badge variant={getScoreBadgeVariant(metrics.precision)}>
                  {formatPercentage(metrics.precision)}%
                </Badge>
              </div>
              <Progress 
                value={(metrics.precision || 0) * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Recall (Sensitivity)</span>
                <Badge variant={getScoreBadgeVariant(metrics.recall)}>
                  {formatPercentage(metrics.recall)}%
                </Badge>
              </div>
              <Progress 
                value={(metrics.recall || 0) * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">F1 Score</span>
                <Badge variant={getScoreBadgeVariant(metrics.f1_score)}>
                  {formatPercentage(metrics.f1_score)}%
                </Badge>
              </div>
              <Progress 
                value={(metrics.f1_score || 0) * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Specificity</span>
                <Badge variant={getScoreBadgeVariant(metrics.specificity)}>
                  {formatPercentage(metrics.specificity)}%
                </Badge>
              </div>
              <Progress 
                value={(metrics.specificity || 0) * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">ROC AUC</span>
                <Badge variant={getScoreBadgeVariant(metrics.roc_auc)}>
                  {formatDecimal(metrics.roc_auc)}
                </Badge>
              </div>
              <Progress 
                value={(metrics.roc_auc || 0) * 100} 
                className="h-2"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Confusion Matrix Details */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Confusion Matrix Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">{metrics.true_positives || 0}</div>
              <div className="text-sm text-green-700">True Positives</div>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="text-2xl font-bold text-red-600">{metrics.false_positives || 0}</div>
              <div className="text-sm text-red-700">False Positives</div>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="text-2xl font-bold text-red-600">{metrics.false_negatives || 0}</div>
              <div className="text-sm text-red-700">False Negatives</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">{metrics.true_negatives || 0}</div>
              <div className="text-sm text-green-700">True Negatives</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Additional Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Additional Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Average Precision</span>
                <Badge variant={getScoreBadgeVariant(metrics.average_precision)}>
                  {formatDecimal(metrics.average_precision)}
                </Badge>
              </div>
              <Progress 
                value={(metrics.average_precision || 0) * 100} 
                className="h-2"
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
