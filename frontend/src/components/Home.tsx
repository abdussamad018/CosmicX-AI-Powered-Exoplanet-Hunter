import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Upload, 
  FlaskConical, 
  Database, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  X
} from 'lucide-react';
import { useState } from 'react';

interface HomeProps {
  onSectionChange: (section: string) => void;
}

export function Home({ onSectionChange }: HomeProps) {
  const [showTips, setShowTips] = useState(true);

  const performanceMetrics = [
    { label: 'Accuracy', value: 0.96, color: 'text-green-600' },
    { label: 'Precision', value: 0.94, color: 'text-blue-600' },
    { label: 'Recall', value: 0.92, color: 'text-purple-600' },
    { label: 'F1 Score', value: 0.93, color: 'text-orange-600' },
    { label: 'ROC-AUC', value: 0.98, color: 'text-red-600' },
  ];

  const recentExperiments = [
    {
      id: 'EXP-2024-001',
      name: 'CNN Transit Detection',
      dataset: 'Kepler Q1-Q8',
      status: 'success',
      accuracy: 0.96,
      timestamp: '2 hours ago'
    },
    {
      id: 'EXP-2024-002', 
      name: '1D-ResNet Fine-tuning',
      dataset: 'TESS Sectors 1-26',
      status: 'running',
      accuracy: 0.89,
      timestamp: '6 hours ago'
    },
    {
      id: 'EXP-2024-003',
      name: 'XGBoost Baseline',
      dataset: 'K2 Campaign 5',
      status: 'failed',
      accuracy: 0.82,
      timestamp: '1 day ago'
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'running':
        return <Clock className="w-4 h-4 text-blue-600" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'success':
        return <Badge variant="default" className="bg-green-100 text-green-800">Success</Badge>;
      case 'running':
        return <Badge variant="default" className="bg-blue-100 text-blue-800">Running</Badge>;
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Hero Section */}
      <div className="text-center py-8">
        <h1 className="mb-4">ExoScope</h1>
        <p className="text-lg text-muted-foreground mb-6 max-w-2xl mx-auto">
          AI-powered platform for discovering exoplanets through machine learning analysis of stellar light curves from NASA's Kepler, K2, and TESS missions.
        </p>
        <Button size="lg" onClick={() => onSectionChange('datasets')}>
          <Database className="w-5 h-5 mr-2" />
          Start with a Dataset
        </Button>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Jump into your workflow</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2" onClick={() => onSectionChange('datasets')}>
              <Upload className="w-6 h-6" />
              <span>Upload CSV/Light Curve</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2" onClick={() => onSectionChange('experiments')}>
              <FlaskConical className="w-6 h-6" />
              <span>Create Experiment</span>
            </Button>
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2" onClick={() => onSectionChange('datasets')}>
              <Database className="w-6 h-6" />
              <span>Import from NASA</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Snapshot */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              Model Performance Snapshot
            </CardTitle>
            <CardDescription>Latest validation metrics across all models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {performanceMetrics.map((metric) => (
              <div key={metric.label} className="flex items-center justify-between">
                <span className="text-sm">{metric.label}</span>
                <div className="flex items-center space-x-3 flex-1 ml-4">
                  <Progress value={metric.value * 100} className="flex-1" />
                  <span className={`text-sm font-medium ${metric.color}`}>
                    {(metric.value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Recent Experiments */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Experiments</CardTitle>
            <CardDescription>Your latest model training runs</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentExperiments.map((exp) => (
                <div key={exp.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-accent/50 cursor-pointer transition-colors" onClick={() => onSectionChange('experiments')}>
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(exp.status)}
                    <div>
                      <div className="font-medium">{exp.name}</div>
                      <div className="text-sm text-muted-foreground">{exp.dataset}</div>
                    </div>
                  </div>
                  <div className="text-right space-y-1">
                    {getStatusBadge(exp.status)}
                    <div className="text-xs text-muted-foreground">{exp.timestamp}</div>
                  </div>
                </div>
              ))}
            </div>
            <Button variant="outline" className="w-full mt-4" onClick={() => onSectionChange('experiments')}>
              View All Experiments
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Tips Panel */}
      {showTips && (
        <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-blue-900 dark:text-blue-100">What's New & Tips</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => setShowTips(false)} className="text-blue-900 dark:text-blue-100 hover:bg-blue-100 dark:hover:bg-blue-900">
              <X className="w-4 h-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
              <li>• New TESS Sector 69 data now available in the catalog</li>
              <li>• Improved phase folding algorithm reduces noise by 15%</li>
              <li>• Try the new explainability features to understand model decisions</li>
              <li>• Pro tip: Use Savitzky-Golay detrending for better transit detection</li>
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}