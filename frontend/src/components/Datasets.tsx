import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from './ui/sheet';
import { Progress } from './ui/progress';
import { 
  Database, 
  Download, 
  Eye, 
  Upload, 
  Filter, 
  Grid, 
  List,
  Calendar,
  HardDrive,
  Star,
  CheckCircle,
  AlertTriangle,
  FileX,
  Loader2
} from 'lucide-react';
import { datasetsApi } from '../lib/api';

// Available NASA missions for download
const availableMissions = [
  { id: 'kepler', name: 'Kepler', description: 'Kepler Mission Exoplanet Candidates' },
  { id: 'k2', name: 'K2', description: 'K2 Mission Extended Data' },
  { id: 'tess', name: 'TESS', description: 'TESS Objects of Interest' }
];

export function Datasets() {
  const [activeTab, setActiveTab] = useState('catalog');
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [missionFilter, setMissionFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMission, setUploadMission] = useState('');
  const [uploadDescription, setUploadDescription] = useState('');
  const [datasets, setDatasets] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await datasetsApi.getDatasets();
      setDatasets(response.datasets || []);
    } catch (err) {
      setError('Failed to load datasets');
      console.error('Error loading datasets:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Filter datasets based on search and mission
  const filteredDatasets = datasets.filter(dataset => {
    const matchesMission = missionFilter === 'all' || dataset.mission === missionFilter;
    const matchesSearch = dataset.filename?.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesMission && matchesSearch;
  });

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadFile(file);
    }
  };

  // Handle dataset upload
  const handleUpload = async () => {
    if (!uploadFile || !uploadMission) return;
    
    try {
      setIsUploading(true);
      setError(null);
      await datasetsApi.uploadDataset(uploadMission, uploadFile);
      setUploadFile(null);
      setUploadMission('');
      setUploadDescription('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      // Reload datasets after successful upload
      await loadDatasets();
    } catch (err) {
      setError('Upload failed');
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle NASA dataset download
  const handleDownloadNASA = async (mission: string) => {
    // For now, just show a message since we don't have NASA download implemented
    alert(`NASA ${mission} dataset download not yet implemented. Please upload your own ${mission} dataset.`);
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadFile(file);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'processing':
        return <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      case 'failed':
        return <FileX className="w-4 h-4 text-red-600" />;
      default:
        return null;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1>Datasets</h1>
        <p className="text-muted-foreground">Explore and manage exoplanet datasets from NASA missions</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="catalog">Catalog</TabsTrigger>
          <TabsTrigger value="import">Import</TabsTrigger>
          <TabsTrigger value="schema">Schema</TabsTrigger>
          <TabsTrigger value="quality">Data Quality</TabsTrigger>
        </TabsList>

        <TabsContent value="catalog" className="space-y-6">
          {/* Filters and Controls */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
                <div className="flex flex-col md:flex-row gap-4 flex-1">
                  <div className="flex-1 max-w-md">
                    <Input
                      placeholder="Search datasets..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full"
                    />
                  </div>
                  <Select value={missionFilter} onValueChange={setMissionFilter}>
                    <SelectTrigger className="w-32">
                      <SelectValue placeholder="Mission" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Missions</SelectItem>
                      <SelectItem value="Kepler">Kepler</SelectItem>
                      <SelectItem value="K2">K2</SelectItem>
                      <SelectItem value="TESS">TESS</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                  >
                    <Grid className="w-4 h-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'table' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode('table')}
                  >
                    <List className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Dataset Grid/Table View */}
          {isLoading ? (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin mr-2" />
                  <span>Loading datasets...</span>
                </div>
              </CardContent>
            </Card>
          ) : error ? (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-center py-8 text-red-600">
                  <AlertTriangle className="w-6 h-6 mr-2" />
                  <span>{error}</span>
                </div>
              </CardContent>
            </Card>
          ) : filteredDatasets.length === 0 ? (
            <Card>
              <CardContent className="pt-6">
                <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                  <Database className="w-12 h-12 mb-4" />
                  <p className="text-lg font-medium mb-2">No datasets found</p>
                  <p>Upload a dataset or download from NASA to get started</p>
                </div>
              </CardContent>
            </Card>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredDatasets.map((dataset, index) => (
                <Card key={index} className="hover:shadow-lg transition-shadow cursor-pointer">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <Badge variant={dataset.mission === 'kepler' ? 'default' : dataset.mission === 'tess' ? 'secondary' : 'outline'}>
                        {dataset.mission.toUpperCase()}
                      </Badge>
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    </div>
                    <CardTitle className="text-lg">{dataset.filename}</CardTitle>
                    <CardDescription>Uploaded dataset</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center text-muted-foreground">
                        <Calendar className="w-4 h-4 mr-2" />
                        {new Date(dataset.created).toLocaleDateString()}
                      </div>
                      <div className="flex items-center text-muted-foreground">
                        <HardDrive className="w-4 h-4 mr-2" />
                        {(dataset.size / 1024 / 1024).toFixed(1)} MB
                      </div>
                      <div className="flex items-center text-muted-foreground">
                        <Star className="w-4 h-4 mr-2" />
                        {dataset.mission.toUpperCase()} Mission
                      </div>
                      <div className="text-muted-foreground">
                        CSV Format
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Sheet>
                        <SheetTrigger asChild>
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="flex-1"
                            onClick={() => setSelectedDataset(dataset)}
                          >
                            <Eye className="w-4 h-4 mr-2" />
                            View
                          </Button>
                        </SheetTrigger>
                        <SheetContent className="w-[400px] sm:w-[540px]">
                          <SheetHeader>
                            <SheetTitle>{selectedDataset?.filename}</SheetTitle>
                            <SheetDescription>Dataset Information</SheetDescription>
                          </SheetHeader>
                          {selectedDataset && (
                            <div className="mt-6 space-y-6">
                              <div>
                                <h4 className="mb-3">Dataset Information</h4>
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Mission:</span>
                                    <Badge variant="outline">{selectedDataset.mission.toUpperCase()}</Badge>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Filename:</span>
                                    <span>{selectedDataset.filename}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Size:</span>
                                    <span>{(selectedDataset.size / 1024 / 1024).toFixed(1)} MB</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Created:</span>
                                    <span>{new Date(selectedDataset.created).toLocaleString()}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">Path:</span>
                                    <span className="text-xs">{selectedDataset.path}</span>
                                  </div>
                                </div>
                              </div>
                              
                              <Button className="w-full">
                                <Download className="w-4 h-4 mr-2" />
                                Use for Training
                              </Button>
                            </div>
                          )}
                        </SheetContent>
                      </Sheet>
                      <Button variant="default" size="sm" className="flex-1">
                        <Download className="w-4 h-4 mr-2" />
                        Use
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Mission</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredDatasets.map((dataset, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <div>
                          <div className="font-medium">{dataset.filename}</div>
                          <div className="text-sm text-muted-foreground">Uploaded dataset</div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={dataset.mission === 'kepler' ? 'default' : dataset.mission === 'tess' ? 'secondary' : 'outline'}>
                          {dataset.mission.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>{(dataset.size / 1024 / 1024).toFixed(1)} MB</TableCell>
                      <TableCell>{new Date(dataset.created).toLocaleDateString()}</TableCell>
                      <TableCell>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            <Eye className="w-4 h-4" />
                          </Button>
                          <Button size="sm">
                            <Download className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="import" className="space-y-6">
          {/* Upload Area */}
          <Card>
            <CardHeader>
              <CardTitle>Import Dataset</CardTitle>
              <CardDescription>Upload CSV files or download NASA datasets</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* File Upload */}
              <div>
                <Label htmlFor="file-upload">Upload File</Label>
                <div 
                  className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.fits,.zip,.tar.gz"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  {uploadFile ? (
                    <div>
                      <h3 className="mb-2 text-green-600">File Selected</h3>
                      <p className="text-sm text-muted-foreground mb-4">{uploadFile.name}</p>
                    </div>
                  ) : (
                    <div>
                      <h3 className="mb-2">Drag and drop files here</h3>
                      <p className="text-muted-foreground mb-4">Supports CSV, FITS, and compressed archives up to 10GB</p>
                    </div>
                  )}
                  <Button variant="outline">Choose Files</Button>
                </div>
              </div>

              {/* Upload Configuration */}
              {uploadFile && (
                <div className="space-y-4 p-4 border rounded-lg bg-muted/50">
                  <h4 className="font-medium">Upload Configuration</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="mission">Mission *</Label>
                      <Select value={uploadMission} onValueChange={setUploadMission}>
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
                      <Label htmlFor="description">Description (Optional)</Label>
                      <Input
                        value={uploadDescription}
                        onChange={(e) => setUploadDescription(e.target.value)}
                        placeholder="Brief description of the dataset"
                      />
                    </div>
                  </div>
                  <Button 
                    onClick={handleUpload} 
                    disabled={isUploading || !uploadMission}
                    className="w-full"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Dataset
                      </>
                    )}
                  </Button>
                  {error && (
                    <div className="text-red-600 text-sm text-center">
                      {error}
                    </div>
                  )}
                </div>
              )}

              {/* NASA Dataset Download */}
              <div className="border-t pt-6">
                <h4 className="font-medium mb-4">Download NASA Datasets</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {availableMissions.map((mission) => (
                    <Card key={mission.id} className="hover:shadow-md transition-shadow">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base">{mission.name}</CardTitle>
                        <CardDescription className="text-sm">{mission.description}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          className="w-full"
                          onClick={() => handleDownloadNASA(mission.id)}
                          disabled={false}
                        >
                          {false ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              Downloading...
                            </>
                          ) : (
                            <>
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </>
                          )}
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recent Datasets */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Datasets</CardTitle>
              <CardDescription>Recently uploaded and processed datasets</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin mr-2" />
                  <span>Loading datasets...</span>
                </div>
              ) : datasets.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                  <Database className="w-12 h-12 mb-4" />
                  <p>No datasets uploaded yet</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Mission</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Created</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {datasets.slice(0, 5).map((dataset, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{dataset.filename}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{dataset.mission.toUpperCase()}</Badge>
                        </TableCell>
                        <TableCell>{(dataset.size / 1024 / 1024).toFixed(1)} MB</TableCell>
                        <TableCell>
                          {new Date(dataset.created).toLocaleDateString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="schema" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Dataset Schema</CardTitle>
              <CardDescription>Standard field definitions and data types</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Database className="w-12 h-12 mx-auto mb-4" />
                <p>Select a dataset from the catalog to view its schema</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Data Quality Assessment</CardTitle>
              <CardDescription>Missing values, outliers, and distribution analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <AlertTriangle className="w-12 h-12 mx-auto mb-4" />
                <p>Load a dataset to view quality metrics and distributions</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}