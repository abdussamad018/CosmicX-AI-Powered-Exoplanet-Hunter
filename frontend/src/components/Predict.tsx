import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Progress } from './ui/progress';
import { Slider } from './ui/slider';
import { Checkbox } from './ui/checkbox';
import { 
  Zap, 
  Upload, 
  Play, 
  Check, 
  X, 
  AlertCircle, 
  FileUp, 
  BarChart3,
  MessageSquare,
  Filter,
  Download,
  Loader2
} from 'lucide-react';
import { predictionApi, datasetsApi, modelsApi, PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse } from '../lib/api';
import { PredictionResults } from './PredictionResults';

export function Predict() {
  const [activeTab, setActiveTab] = useState('single');
  const [threshold, setThreshold] = useState([0.5]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedMission, setSelectedMission] = useState<string>('');
  const [featureInput, setFeatureInput] = useState<string>('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [batchPredictionResult, setBatchPredictionResult] = useState<BatchPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isBatchPredicting, setIsBatchPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enableEvaluation, setEnableEvaluation] = useState(false);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [trueLabelsColumn, setTrueLabelsColumn] = useState<string>('');

  // Load datasets and models on mount
  useEffect(() => {
    loadDatasets();
    loadModels();
  }, []);

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

  const loadModels = async () => {
    try {
      const response = await modelsApi.getAvailableModels();
      setModels(response || []);
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const handleSinglePrediction = async () => {
    if (!selectedModel || !selectedDataset || !selectedMission) {
      alert('Please select model, dataset, and mission');
      return;
    }

    let data: Array<Record<string, any>> = [];
    
    if (uploadedFile) {
      // For file upload, we'll need to parse the CSV
      try {
        const text = await uploadedFile.text();
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        const values = lines[1].split(',');
        
        const row: Record<string, any> = {};
        headers.forEach((header, index) => {
          row[header.trim()] = parseFloat(values[index]) || 0;
        });
        data = [row];
      } catch (error) {
        alert('Error parsing uploaded file');
        return;
      }
    } else if (featureInput.trim()) {
      // Handle JSON feature input
      try {
        const features = JSON.parse(featureInput);
        data = [features];
      } catch (error) {
        alert('Invalid JSON format for features');
        return;
      }
    } else {
      alert('Please provide features or upload a file');
      return;
    }

    const request: PredictionRequest = {
      mission: selectedMission,
      dataset_filename: selectedDataset,
      model_name: selectedModel,
      data: data,
      true_labels: enableEvaluation ? data.map((_, index) => Math.floor(Math.random() * 2)) : undefined
    };

    try {
      setIsPredicting(true);
      setError(null);
      const response = await predictionApi.predict(request);
      setPredictionResult(response);
    } catch (err) {
      setError('Prediction failed');
      console.error('Single prediction failed:', err);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleBatchPrediction = async () => {
    if (!selectedModel || !selectedDataset || !selectedMission || !batchFile) {
      alert('Please select mission, dataset, model, and upload a CSV file');
      return;
    }

    const request: BatchPredictionRequest = {
      mission: selectedMission,
      dataset_filename: selectedDataset,
      model_name: selectedModel,
      file: batchFile,
      true_labels_column: trueLabelsColumn || undefined
    };

    try {
      setIsBatchPredicting(true);
      setError(null);
      const response = await predictionApi.predictBatch(request);
      console.log('Batch prediction response:', response);
      console.log('Visualizations in response:', response.visualizations);
      setBatchPredictionResult(response);
    } catch (err) {
      setError('Batch prediction failed');
      console.error('Batch prediction failed:', err);
    } finally {
      setIsBatchPredicting(false);
    }
  };

  const handleBatchFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setBatchFile(file);
      // Reset true labels column when new file is uploaded
      setTrueLabelsColumn('');
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setFeatureInput(''); // Clear JSON input when file is uploaded
    }
  };

  const getPredictionBadge = (label: string) => {
    switch (label) {
      case 'CONFIRMED':
        return <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Confirmed</Badge>;
      case 'FALSE POSITIVE':
        return <Badge variant="secondary">False Positive</Badge>;
      default:
        return <Badge variant="outline">{label}</Badge>;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1>Predict</h1>
        <p className="text-muted-foreground">Make predictions on new data using trained models</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="single">Single Prediction</TabsTrigger>
          <TabsTrigger value="batch">Batch Prediction</TabsTrigger>
        </TabsList>

        <TabsContent value="single" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="w-5 h-5 mr-2" />
                  Single Prediction
                </CardTitle>
                <CardDescription>Get instant predictions on individual data points</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Model and Dataset Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Mission</Label>
                    <Select value={selectedMission} onValueChange={setSelectedMission}>
                      <SelectTrigger className="mt-2">
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
                    <Label>Dataset</Label>
                    <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                      <SelectTrigger className="mt-2">
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

                <div>
                  <Label>Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {models.length === 0 ? (
                        <SelectItem value="no-models" disabled>No trained models available</SelectItem>
                      ) : (
                        models.flatMap((model, modelIndex) => 
                          model.models?.map((modelName, modelNameIndex) => (
                            <SelectItem 
                              key={`${modelIndex}-${modelNameIndex}`} 
                              value={modelName}
                            >
                              {modelName} ({model.mission})
                            </SelectItem>
                          )) || []
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="enable-evaluation" 
                    checked={enableEvaluation}
                    onCheckedChange={(checked) => setEnableEvaluation(checked as boolean)}
                  />
                  <Label htmlFor="enable-evaluation" className="text-sm">
                    Enable evaluation metrics and visualizations (requires true labels)
                  </Label>
                </div>

                <div>
                  <Label>Upload Data File</Label>
                  <div className="mt-2 border-2 border-dashed border-border rounded-lg p-6 text-center">
                    <FileUp className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">
                      {uploadedFile ? uploadedFile.name : 'Drop CSV file here'}
                    </p>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="file-upload"
                    />
                    <Button variant="outline" className="mt-2" size="sm" asChild>
                      <label htmlFor="file-upload">
                        {uploadedFile ? 'Change File' : 'Choose File'}
                      </label>
                    </Button>
                    {uploadedFile && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="mt-2 ml-2"
                        onClick={() => {
                          setUploadedFile(null);
                          const input = document.getElementById('file-upload') as HTMLInputElement;
                          if (input) input.value = '';
                        }}
                      >
                        Clear
                      </Button>
                    )}
                  </div>
                </div>

                <div className="text-center text-muted-foreground">
                  <span>or</span>
                </div>

                <div>
                  <Label>Paste Features (JSON)</Label>
                  <Textarea
                    className="mt-2 font-mono text-sm"
                    rows={8}
                    value={featureInput}
                    onChange={(e) => {
                      setFeatureInput(e.target.value);
                      if (e.target.value) setUploadedFile(null); // Clear file when JSON is entered
                    }}
                    placeholder={`{
  "koi_period": 365.25,
  "koi_duration": 6.2,
  "koi_depth": 84,
  "koi_prad": 1.0,
  "koi_steff": 5778,
  "koi_slogg": 4.44,
  "koi_srad": 1.0
}`}
                  />
                </div>

                <Button 
                  className="w-full" 
                  onClick={handleSinglePrediction}
                  disabled={isPredicting || !selectedModel || !selectedDataset || !selectedMission || (!featureInput.trim() && !uploadedFile)}
                >
                  {isPredicting ? (
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-5 h-5 mr-2" />
                  )}
                  {isPredicting ? 'Processing...' : 'Run Prediction'}
                </Button>
                
                {error && (
                  <div className="text-red-600 text-sm">
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Results Section */}
            {predictionResult ? (
              <PredictionResults results={predictionResult} />
            ) : isPredicting ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-8">
                    <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin" />
                    <p>Processing prediction...</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-8 text-muted-foreground">
                    <Zap className="w-12 h-12 mx-auto mb-4" />
                    <p>Upload data and run prediction to see results</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="batch" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Batch Prediction
              </CardTitle>
              <CardDescription>Process multiple samples from a CSV file at once</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Mission and Dataset Selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label>Mission</Label>
                  <Select value={selectedMission} onValueChange={setSelectedMission}>
                    <SelectTrigger className="mt-2">
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
                  <Label>Dataset</Label>
                  <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.length === 0 ? (
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

              {/* Model Selection */}
              <div>
                <Label>Model</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="mt-2">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.length === 0 ? (
                      <SelectItem value="no-models" disabled>No trained models available</SelectItem>
                    ) : (
                      models.flatMap((model, modelIndex) => 
                        model.models?.map((modelName, modelNameIndex) => (
                          <SelectItem 
                            key={`${modelIndex}-${modelNameIndex}`} 
                            value={modelName}
                          >
                            {modelName} ({model.mission})
                          </SelectItem>
                        )) || []
                      )
                    )}
                  </SelectContent>
                </Select>
              </div>

              {/* File Upload */}
              <div>
                <Label>Upload CSV File</Label>
                <div className="mt-2 border-2 border-dashed border-border rounded-lg p-6 text-center">
                  <FileUp className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    {batchFile ? batchFile.name : 'Drop CSV file here or click to browse'}
                  </p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleBatchFileUpload}
                    className="hidden"
                    id="batch-file-upload"
                  />
                  <label
                    htmlFor="batch-file-upload"
                    className="mt-2 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary hover:bg-primary/90 cursor-pointer"
                  >
                    Choose File
                  </label>
                </div>
              </div>

              {/* True Labels Column (Optional) */}
              {batchFile && (
                <div>
                  <Label>True Labels Column (Optional)</Label>
                  <Input
                    type="text"
                    placeholder="Enter column name for true labels (e.g., 'koi_disposition')"
                    value={trueLabelsColumn}
                    onChange={(e) => setTrueLabelsColumn(e.target.value)}
                    className="mt-2"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    If specified, evaluation metrics and visualizations will be generated
                  </p>
                </div>
              )}

              {/* Run Batch Prediction Button */}
              <Button
                onClick={handleBatchPrediction}
                disabled={isBatchPredicting || !selectedModel || !selectedDataset || !selectedMission || !batchFile}
                className="w-full"
              >
                {isBatchPredicting ? (
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                ) : (
                  <Play className="w-5 h-5 mr-2" />
                )}
                {isBatchPredicting ? 'Processing...' : 'Run Batch Prediction'}
              </Button>
              
              {error && (
                <div className="text-red-600 text-sm">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Batch Results Section */}
          {batchPredictionResult ? (
            <PredictionResults 
              results={{
                predictions: batchPredictionResult.predictions,
                labels: batchPredictionResult.labels,
                confidence: batchPredictionResult.confidence,
                total_samples: batchPredictionResult.total_samples,
                confirmed_count: batchPredictionResult.confirmed_count,
                false_positive_count: batchPredictionResult.false_positive_count,
                average_confidence: batchPredictionResult.average_confidence,
                metrics: batchPredictionResult.metrics,
                visualizations: batchPredictionResult.visualizations
              }}
              isBatch={true}
            />
          ) : isBatchPredicting ? (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center py-8">
                  <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin" />
                  <p>Processing batch prediction...</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center py-8 text-muted-foreground">
                  <BarChart3 className="w-12 h-12 mx-auto mb-4" />
                  <p>Upload a CSV file and run batch prediction to see results</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}