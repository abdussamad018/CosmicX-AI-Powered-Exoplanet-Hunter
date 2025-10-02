import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Slider } from './ui/slider';
import { Progress } from './ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { 
  FlaskConical, 
  Play, 
  Pause, 
  Square, 
  Brain, 
  Cpu, 
  BarChart3, 
  Eye, 
  GitBranch,
  Settings,
  Target,
  TrendingUp,
  Download,
  Zap,
  Loader2
} from 'lucide-react';
import { trainingApi, datasetsApi, modelsApi, TrainingRequest, TrainingStatus } from '../lib/api';

const modelTypes = [
  {
    id: 'cnn1d',
    name: '1D CNN',
    description: 'Convolutional neural network for time series',
    params: { layers: 4, filters: 64, kernelSize: 3 }
  },
  {
    id: 'resnet1d',
    name: '1D ResNet',
    description: 'Residual network with skip connections',
    params: { blocks: 3, filters: 128, depth: 18 }
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    description: 'Gradient boosting for tabular features',
    params: { estimators: 100, depth: 6, learning_rate: 0.1 }
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: 'Attention-based architecture',
    params: { heads: 8, layers: 6, dim: 256 }
  }
];

export function ModelLab() {
  const [activeTab, setActiveTab] = useState('configure');
  const [selectedModels, setSelectedModels] = useState<string[]>(['xgboost']);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedMission, setSelectedMission] = useState<string>('');
  const [testSize, setTestSize] = useState([0.2]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingStatus[]>([]);
  const [currentTraining, setCurrentTraining] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load datasets and training history on mount
  useEffect(() => {
    loadDatasets();
    loadTrainingHistory();
  }, []);

  // Poll training status if there's an active training
  useEffect(() => {
    let interval: number;
    if (currentTraining && currentTraining.status === 'training') {
      interval = setInterval(async () => {
        try {
          const status = await trainingApi.getTrainingStatus(currentTraining.training_id);
          setCurrentTraining(status);
          if (status.status === 'completed' || status.status === 'failed') {
            setIsTraining(false);
            loadTrainingHistory(); // Refresh history
          }
        } catch (err) {
          console.error('Error polling training status:', err);
        }
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentTraining]);

  const loadDatasets = async () => {
    try {
      setIsLoading(true);
      const response = await datasetsApi.getDatasets();
      setDatasets(response.datasets || []);
    } catch (err) {
      setError('Failed to load datasets');
      console.error('Error loading datasets:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadTrainingHistory = async () => {
    try {
      const history = await trainingApi.getTrainingHistory();
      setTrainingHistory(history);
    } catch (err) {
      console.error('Error loading training history:', err);
    }
  };

  const toggleModel = (modelId: string) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else {
        return [...prev, modelId];
      }
    });
  };

  const handleStartTraining = async () => {
    if (!selectedDataset || !selectedMission) {
      alert('Please select a dataset and mission');
      return;
    }

    if (selectedModels.length === 0) {
      alert('Please select at least one model');
      return;
    }

    try {
      setIsTraining(true);
      setError(null);
      
      const request: TrainingRequest = {
        mission: selectedMission,
        dataset_filename: selectedDataset,
        models: selectedModels,
        test_size: testSize[0]
      };

      const response = await trainingApi.startTraining(request);
      
      // Start polling for status
      const status = await trainingApi.getTrainingStatus(response.training_id);
      setCurrentTraining(status);
      
      setActiveTab('training');
    } catch (err) {
      setError('Failed to start training');
      console.error('Failed to start training:', err);
      setIsTraining(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <div className="w-2 h-2 bg-green-500 rounded-full" />;
      case 'training':
      case 'loading_data':
      case 'saving':
        return <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />;
      case 'failed':
        return <div className="w-2 h-2 bg-red-500 rounded-full" />;
      default:
        return null;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1>Model Lab</h1>
        <p className="text-muted-foreground">Train and evaluate machine learning models for exoplanet detection</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="configure">Configure</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="validation">Validation</TabsTrigger>
          <TabsTrigger value="explainability">Explainability</TabsTrigger>
          <TabsTrigger value="compare">Compare</TabsTrigger>
        </TabsList>

        <TabsContent value="configure" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Model Selection */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="w-5 h-5 mr-2" />
                  Model Architecture
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {modelTypes.map((model) => (
                  <div
                    key={model.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedModels.includes(model.id)
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:bg-accent/50'
                    }`}
                    onClick={() => toggleModel(model.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-medium">{model.name}</h4>
                        <p className="text-sm text-muted-foreground">{model.description}</p>
                      </div>
                      {selectedModels.includes(model.id) && (
                        <div className="w-4 h-4 bg-primary rounded-full flex items-center justify-center">
                          <div className="w-2 h-2 bg-white rounded-full" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div className="text-xs text-muted-foreground">
                  {selectedModels.length} model(s) selected
                </div>
              </CardContent>
            </Card>

            {/* Configuration Panel */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Dataset Selection */}
                <div>
                  <h4 className="mb-4">Dataset</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label>Mission</Label>
                      <Select value={selectedMission} onValueChange={setSelectedMission}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select mission" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="kepler">Kepler</SelectItem>
                          <SelectItem value="k2">K2</SelectItem>
                          <SelectItem value="tess">TESS</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Dataset File</Label>
                      <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select dataset" />
                        </SelectTrigger>
                        <SelectContent>
                          {isLoading ? (
                            <SelectItem value="loading" disabled>Loading datasets...</SelectItem>
                          ) : datasets.length === 0 ? (
                            <SelectItem value="no-data" disabled>No datasets available</SelectItem>
                          ) : (
                            datasets
                              .filter(d => !selectedMission || d.mission === selectedMission)
                              .map((dataset, index) => (
                                <SelectItem key={index} value={dataset.filename}>
                                  {dataset.filename}
                                </SelectItem>
                              ))
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                {/* Test Size */}
                <div>
                  <h4 className="mb-4">Test Split</h4>
                  <div className="space-y-2">
                    <Label>Test Size: {(testSize[0] * 100).toFixed(0)}%</Label>
                    <Slider
                      value={testSize}
                      onValueChange={setTestSize}
                      max={0.5}
                      min={0.1}
                      step={0.05}
                    />
                    <div className="text-xs text-muted-foreground">
                      Train: {((1 - testSize[0]) * 100).toFixed(0)}% / Test: {(testSize[0] * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-4 pt-4">
                  <Button 
                    onClick={handleStartTraining} 
                    size="lg"
                    disabled={isTraining || !selectedDataset || !selectedMission || selectedModels.length === 0}
                  >
                    {isTraining ? (
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    ) : (
                      <Play className="w-5 h-5 mr-2" />
                    )}
                    {isTraining ? 'Starting...' : 'Start Training'}
                  </Button>
                </div>
                {error && (
                  <div className="text-red-600 text-sm">
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="training" className="space-y-6">
          {/* Training Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Cpu className="w-5 h-5 mr-2" />
                Training Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {currentTraining ? (
                  <>
                    <div className="flex items-center justify-between">
                      <span>Progress</span>
                      <span>{currentTraining.progress.toFixed(1)}%</span>
                    </div>
                    <Progress value={currentTraining.progress} />
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Status</div>
                        <div className="font-medium capitalize">{currentTraining.status}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Mission</div>
                        <div className="font-medium">{currentTraining.mission}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Models</div>
                        <div className="font-medium">{currentTraining.models.join(', ')}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Progress</div>
                        <div className="font-medium">{currentTraining.progress.toFixed(1)}%</div>
                      </div>
                    </div>
                    
                    {currentTraining.error && (
                      <div className="text-red-600 text-sm p-3 bg-red-50 rounded">
                        Error: {currentTraining.error}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Cpu className="w-12 h-12 mx-auto mb-4" />
                    <p>No active training session</p>
                    <p className="text-sm">Start a training job to see progress here</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="validation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Validation</CardTitle>
              <CardDescription>View training results and metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <BarChart3 className="w-12 h-12 mx-auto mb-4" />
                <p>Complete training to view validation metrics</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="explainability" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Eye className="w-5 h-5 mr-2" />
                Model Explainability
              </CardTitle>
              <CardDescription>
                Understand what the model learned and which features are most important
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Target className="w-12 h-12 mx-auto mb-4" />
                <p>SHAP explanations and feature importance</p>
                <p className="text-sm">Complete training to view model explanations</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compare" className="space-y-6">
          {/* Experiment Comparison */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <GitBranch className="w-5 h-5 mr-2" />
                Training History
              </CardTitle>
              <CardDescription>Compare metrics across different training runs</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Training ID</TableHead>
                    <TableHead>Mission</TableHead>
                    <TableHead>Models</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Progress</TableHead>
                    <TableHead>Start Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trainingHistory.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                        No training runs found
                      </TableCell>
                    </TableRow>
                  ) : (
                    trainingHistory.map((run) => (
                      <TableRow key={run.training_id}>
                        <TableCell className="font-mono text-xs">{run.training_id}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{run.mission}</Badge>
                        </TableCell>
                        <TableCell>{run.models.join(', ')}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(run.status)}
                            <span className="capitalize">{run.status}</span>
                          </div>
                        </TableCell>
                        <TableCell>{run.progress.toFixed(1)}%</TableCell>
                        <TableCell>
                          {run.start_time ? new Date(run.start_time).toLocaleString() : '-'}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}