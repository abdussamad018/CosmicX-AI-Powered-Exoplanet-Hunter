import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface VisualizationDisplayProps {
  visualizations: {
    roc_curve?: string;
    pr_curve?: string;
    confusion_matrix?: string;
    shap_importance?: string;
  };
}

export const VisualizationDisplay: React.FC<VisualizationDisplayProps> = ({ visualizations }) => {
  const hasVisualizations = Object.values(visualizations).some(viz => viz && viz.length > 0);

  if (!hasVisualizations) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Model Visualizations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <p>No visualizations available. Provide true labels to generate evaluation charts.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg font-semibold">Model Visualizations</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="roc" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="roc">ROC Curve</TabsTrigger>
            <TabsTrigger value="pr">Precision-Recall</TabsTrigger>
            <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
            <TabsTrigger value="shap">SHAP Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="roc" className="mt-4">
            {visualizations.roc_curve ? (
              <div className="flex justify-center">
                <img 
                  src={visualizations.roc_curve} 
                  alt="ROC Curve" 
                  className="max-w-full h-auto rounded-lg border shadow-sm"
                />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>ROC curve not available</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="pr" className="mt-4">
            {visualizations.pr_curve ? (
              <div className="flex justify-center">
                <img 
                  src={visualizations.pr_curve} 
                  alt="Precision-Recall Curve" 
                  className="max-w-full h-auto rounded-lg border shadow-sm"
                />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Precision-Recall curve not available</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="confusion" className="mt-4">
            {visualizations.confusion_matrix ? (
              <div className="flex justify-center">
                <img 
                  src={visualizations.confusion_matrix} 
                  alt="Confusion Matrix" 
                  className="max-w-full h-auto rounded-lg border shadow-sm"
                />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Confusion matrix not available</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="shap" className="mt-4">
            {visualizations.shap_importance ? (
              <div className="flex justify-center">
                <img 
                  src={visualizations.shap_importance} 
                  alt="SHAP Feature Importance" 
                  className="max-w-full h-auto rounded-lg border shadow-sm"
                />
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>SHAP analysis not available (only supported for XGBoost models)</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
