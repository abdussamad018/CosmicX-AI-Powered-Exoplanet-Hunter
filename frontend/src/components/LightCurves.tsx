import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Switch } from './ui/switch';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Separator } from './ui/separator';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { 
  Activity, 
  Settings, 
  Play, 
  Download, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  Waves,
  Filter,
  Save,
  Layout
} from 'lucide-react';

const mockLightCurveData = [
  { time: 0, flux: 1.002 },
  { time: 1, flux: 1.001 },
  { time: 2, flux: 0.999 },
  { time: 3, flux: 0.997 },
  { time: 4, flux: 0.995 },
  { time: 5, flux: 0.998 },
  { time: 6, flux: 1.001 },
  { time: 7, flux: 1.003 },
  { time: 8, flux: 1.000 },
  { time: 9, flux: 0.998 },
];

const starMetadata = {
  name: 'Kepler-452',
  keplerMag: 13.426,
  stellarRadius: 1.11,
  stellarMass: 1.04,
  teff: 5757,
  ra: 292.4849,
  dec: 44.2793,
  distance: 1402
};

const preprocessingSteps = [
  { id: 'normalize', name: 'Normalize', description: 'Scale flux to relative values', enabled: true },
  { id: 'detrend', name: 'Detrend (Savitzky-Golay)', description: 'Remove instrumental trends', enabled: false },
  { id: 'outliers', name: 'Remove Outliers', description: 'Filter statistical outliers', enabled: false },
  { id: 'fold', name: 'Phase Fold', description: 'Fold to orbital period', enabled: false },
  { id: 'bin', name: 'Bin Data', description: 'Reduce noise by binning', enabled: false }
];

const prebuiltTemplates = [
  { name: 'Kepler Classic', description: 'Standard preprocessing for Kepler data', steps: ['normalize', 'detrend'] },
  { name: 'TESS SNR-Boost', description: 'Noise reduction for TESS observations', steps: ['normalize', 'detrend', 'bin'] },
  { name: 'Transit Search', description: 'Optimized for transit detection', steps: ['normalize', 'detrend', 'outliers', 'fold'] }
];

export function LightCurves() {
  const [activeTab, setActiveTab] = useState('viewer');
  const [detrendEnabled, setDetrendEnabled] = useState(false);
  const [normalizeEnabled, setNormalizeEnabled] = useState(true);
  const [phaseFoldEnabled, setPhaseFoldEnabled] = useState(false);
  const [period, setPeriod] = useState([368.865]);
  const [cadence, setCadence] = useState('30min');
  const [activeSteps, setActiveSteps] = useState(new Set(['normalize']));

  const toggleStep = (stepId: string) => {
    const newSteps = new Set(activeSteps);
    if (newSteps.has(stepId)) {
      newSteps.delete(stepId);
    } else {
      newSteps.add(stepId);
    }
    setActiveSteps(newSteps);
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1>Light Curves</h1>
        <p className="text-muted-foreground">Visualize and preprocess stellar light curves for exoplanet detection</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="viewer">Viewer</TabsTrigger>
          <TabsTrigger value="preprocess">Preprocess</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
        </TabsList>

        <TabsContent value="viewer" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Star Metadata Panel */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Activity className="w-5 h-5 mr-2" />
                  Star Metadata
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4>{starMetadata.name}</h4>
                  <Badge variant="outline" className="mt-1">Kepler Target</Badge>
                </div>
                
                <Separator />
                
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Kepler Mag:</span>
                    <span>{starMetadata.keplerMag}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Stellar Radius:</span>
                    <span>{starMetadata.stellarRadius} R☉</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Stellar Mass:</span>
                    <span>{starMetadata.stellarMass} M☉</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">T_eff:</span>
                    <span>{starMetadata.teff} K</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Distance:</span>
                    <span>{starMetadata.distance} ly</span>
                  </div>
                </div>

                <Separator />

                {/* Controls */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Normalize</label>
                    <Switch checked={normalizeEnabled} onCheckedChange={setNormalizeEnabled} />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Detrend</label>
                    <Switch checked={detrendEnabled} onCheckedChange={setDetrendEnabled} />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Phase Fold</label>
                    <Switch checked={phaseFoldEnabled} onCheckedChange={setPhaseFoldEnabled} />
                  </div>

                  {phaseFoldEnabled && (
                    <div className="space-y-2">
                      <label className="text-sm text-muted-foreground">Period (days)</label>
                      <Slider
                        value={period}
                        onValueChange={setPeriod}
                        max={1000}
                        min={0.1}
                        step={0.1}
                        className="w-full"
                      />
                      <div className="text-xs text-muted-foreground text-center">
                        {period[0]} days
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <label className="text-sm text-muted-foreground">Cadence</label>
                    <Select value={cadence} onValueChange={setCadence}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="30min">30 minute</SelectItem>
                        <SelectItem value="1hour">1 hour</SelectItem>
                        <SelectItem value="6hour">6 hour</SelectItem>
                        <SelectItem value="1day">1 day</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Chart Area */}
            <Card className="lg:col-span-3">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Light Curve - {starMetadata.name}</CardTitle>
                  <CardDescription>
                    Time vs Normalized Flux
                    {detrendEnabled && ' (Detrended)'}
                    {phaseFoldEnabled && ' (Phase Folded)'}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    <ZoomIn className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <ZoomOut className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {/* Placeholder for chart - in a real app this would be a proper chart component */}
                <div className="h-96 bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
                  <ImageWithFallback 
                    src="https://images.unsplash.com/photo-1532679170412-b56d03dcd115?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsaWdodCUyMGN1cnZlJTIwY2hhcnQlMjBncmFwaCUyMGFzdHJvbm9teXxlbnwxfHx8fDE3NTgxNjYyMTh8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                    alt="Light curve visualization"
                    className="w-full h-full object-cover opacity-70"
                  />
                  <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
                    <div className="text-center">
                      <Waves className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                      <h3 className="mb-2">Interactive Light Curve</h3>
                      <p className="text-muted-foreground">Pan, zoom, and brush to explore the data</p>
                      <div className="mt-4 text-sm text-muted-foreground">
                        {mockLightCurveData.length} data points • {cadence} cadence
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Chart Controls */}
                <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                  <div>Time (BJD - 2454833)</div>
                  <div>Relative Flux</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="preprocess" className="space-y-6">
          {/* Pipeline Builder */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Preprocessing Pipeline
              </CardTitle>
              <CardDescription>Build a custom preprocessing pipeline for your light curves</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {preprocessingSteps.map((step) => (
                  <div key={step.id} className="flex items-center justify-between p-4 border border-border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Switch 
                        checked={activeSteps.has(step.id)}
                        onCheckedChange={() => toggleStep(step.id)}
                      />
                      <div>
                        <h4 className="font-medium">{step.name}</h4>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                      </div>
                    </div>
                    <Button variant="outline" size="sm">
                      <Settings className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
              
              <div className="flex gap-4 mt-6">
                <Button>
                  <Play className="w-4 h-4 mr-2" />
                  Run Pipeline
                </Button>
                <Button variant="outline">
                  <Save className="w-4 h-4 mr-2" />
                  Save as Template
                </Button>
                <Button variant="outline">
                  <Download className="w-4 h-4 mr-2" />
                  Export Config
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Pipeline Preview</CardTitle>
              <CardDescription>Preview the effect of each preprocessing step</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 bg-muted rounded-lg flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <Filter className="w-12 h-12 mx-auto mb-4" />
                  <p>Run pipeline to see preprocessing effects</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="templates" className="space-y-6">
          {/* Prebuilt Templates */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {prebuiltTemplates.map((template, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow cursor-pointer">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Layout className="w-5 h-5 mr-2" />
                    {template.name}
                  </CardTitle>
                  <CardDescription>{template.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-sm font-medium mb-2">Included Steps:</h4>
                      <div className="space-y-1">
                        {template.steps.map((stepId) => {
                          const step = preprocessingSteps.find(s => s.id === stepId);
                          return step ? (
                            <Badge key={stepId} variant="outline" className="mr-2">
                              {step.name}
                            </Badge>
                          ) : null;
                        })}
                      </div>
                    </div>
                    <Button className="w-full">
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Custom Templates */}
          <Card>
            <CardHeader>
              <CardTitle>My Templates</CardTitle>
              <CardDescription>Your saved preprocessing pipelines</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Layout className="w-12 h-12 mx-auto mb-4" />
                <p>No custom templates saved yet</p>
                <p className="text-sm">Create templates from the Preprocess tab</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}