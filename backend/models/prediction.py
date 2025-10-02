import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import joblib
warnings.filterwarnings("ignore")

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention

# Visualization and metrics imports
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
import shap

class PredictionManager:
    def __init__(self):
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def predict(
        self, 
        mission: str, 
        dataset_filename: str, 
        model_name: str, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            # Find the model
            model_path = await self._find_model(mission, dataset_filename, model_name)
            if not model_path:
                raise ValueError(f"Model not found for mission={mission}, dataset={dataset_filename}, model={model_name}")
            
            # Load model and metadata
            model, metadata, imputer, scaler = await self._load_model(model_path, model_name)
            
            # Prepare input data
            X_pred = self._prepare_prediction_data(data, metadata["feature_names"], imputer, scaler)
            
            # Make predictions
            if model_name == "xgboost":
                dmat = xgb.DMatrix(X_pred)
                predictions = model.predict(dmat)
            else:
                # For deep learning models, add channel dimension
                X_pred_seq = X_pred.values[..., None]
                predictions = model.predict(X_pred_seq, verbose=0).ravel()
            
            # Convert predictions to labels and confidence
            labels = ["CONFIRMED" if p > 0.5 else "FALSE POSITIVE" for p in predictions]
            confidence = [float(p) for p in predictions]
            
            return {
                "predictions": [float(p) for p in predictions],
                "labels": labels,
                "confidence": confidence
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    async def predict_with_evaluation(
        self, 
        mission: str, 
        dataset_filename: str, 
        model_name: str, 
        data: List[Dict[str, Any]],
        true_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Make predictions with comprehensive evaluation metrics and visualizations"""
        try:
            # Find the model
            model_path = await self._find_model(mission, dataset_filename, model_name)
            if not model_path:
                raise ValueError(f"Model not found for mission={mission}, dataset={dataset_filename}, model={model_name}")
            
            # Load model and metadata
            model, metadata, imputer, scaler = await self._load_model(model_path, model_name)
            
            # Prepare input data
            X_pred = self._prepare_prediction_data(data, metadata["feature_names"], imputer, scaler)
            
            # Make predictions
            if model_name == "xgboost":
                dmat = xgb.DMatrix(X_pred)
                predictions = model.predict(dmat)
            else:
                # For deep learning models, add channel dimension
                X_pred_seq = X_pred.values[..., None]
                predictions = model.predict(X_pred_seq, verbose=0).ravel()
            
            # Convert predictions to labels and confidence
            labels = ["CONFIRMED" if p > 0.5 else "FALSE POSITIVE" for p in predictions]
            confidence = [float(p) for p in predictions]
            
            result = {
                "predictions": [float(p) for p in predictions],
                "labels": labels,
                "confidence": confidence
            }
            
            # If true labels are provided, calculate comprehensive metrics
            if true_labels is not None:
                y_true = np.array(true_labels)
                y_pred_binary = (np.array(predictions) > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(y_true, y_pred_binary, predictions)
                result["metrics"] = metrics
                
                # Generate visualizations
                visualizations = await self._generate_visualizations(
                    y_true, predictions, X_pred, model, model_name, metadata
                )
                result["visualizations"] = visualizations
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction with evaluation failed: {str(e)}")
    
    def _calculate_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_binary: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except:
                roc_auc = 0.0
            
            # Average Precision (PR AUC)
            try:
                avg_precision = average_precision_score(y_true, y_pred_proba)
            except:
                avg_precision = 0.0
            
            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "specificity": float(specificity),
                "roc_auc": float(roc_auc),
                "average_precision": float(avg_precision),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn)
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "specificity": 0.0,
                "roc_auc": 0.0,
                "average_precision": 0.0,
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
    
    async def _generate_visualizations(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        X_pred: pd.DataFrame, 
        model, 
        model_name: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive visualizations"""
        try:
            visualizations = {}
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. ROC Curve
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                visualizations["roc_curve"] = self._fig_to_base64()
                plt.close()
            except Exception as e:
                print(f"Error generating ROC curve: {str(e)}")
                visualizations["roc_curve"] = ""
            
            # 2. Precision-Recall Curve
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                avg_precision = average_precision_score(y_true, y_pred_proba)
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='darkorange', lw=2,
                        label=f'PR curve (AP = {avg_precision:.2f})')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, alpha=0.3)
                
                visualizations["pr_curve"] = self._fig_to_base64()
                plt.close()
            except Exception as e:
                print(f"Error generating PR curve: {str(e)}")
                visualizations["pr_curve"] = ""
            
            # 3. Confusion Matrix
            try:
                y_pred_binary = (y_pred_proba > 0.5).astype(int)
                cm = confusion_matrix(y_true, y_pred_binary)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['FALSE POSITIVE', 'CONFIRMED'],
                           yticklabels=['FALSE POSITIVE', 'CONFIRMED'])
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                visualizations["confusion_matrix"] = self._fig_to_base64()
                plt.close()
            except Exception as e:
                print(f"Error generating confusion matrix: {str(e)}")
                visualizations["confusion_matrix"] = ""
            
            # 4. SHAP Feature Importance (for XGBoost)
            if model_name == "xgboost" and len(X_pred) > 0:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_pred.iloc[:min(100, len(X_pred))])
                    
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_pred.iloc[:min(100, len(X_pred))], 
                                    show=False, max_display=20)
                    plt.title('SHAP Feature Importance')
                    plt.tight_layout()
                    
                    visualizations["shap_importance"] = self._fig_to_base64()
                    plt.close()
                except Exception as e:
                    print(f"Error generating SHAP plot: {str(e)}")
                    visualizations["shap_importance"] = ""
            
            return visualizations
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return {}
    
    def _fig_to_base64(self) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error converting figure to base64: {str(e)}")
            return ""
    
    async def _find_model(self, mission: str, dataset_filename: str, model_name: str) -> Optional[Path]:
        """Find the model file for given parameters"""
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                continue
                
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                if (metadata.get("mission") == mission and 
                    model_name in metadata.get("models", [])):
                    return model_dir
            except:
                continue
        
        return None
    
    async def _load_model(self, model_path: Path, model_name: str):
        """Load model and metadata"""
        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load preprocessing objects
        imputer = None
        scaler = None
        try:
            imputer = joblib.load(model_path / "imputer.pkl")
            scaler = joblib.load(model_path / "scaler.pkl")
            print("Loaded imputer and scaler successfully")
        except Exception as e:
            print(f"Warning: Could not load preprocessing objects: {str(e)}")
        
        # Load model
        if model_name == "xgboost":
            model = xgb.Booster()
            model.load_model(str(model_path / f"{model_name}.json"))
        else:
            model = tf.keras.models.load_model(str(model_path / f"{model_name}.h5"))
        
        return model, metadata, imputer, scaler
    
    def _prepare_prediction_data(self, data: List[Dict[str, Any]], feature_names: List[str], imputer=None, scaler=None) -> pd.DataFrame:
        """Prepare input data for prediction"""
        df = pd.DataFrame(data)
        print(f"Input data shape: {df.shape}")
        print(f"Input data columns: {list(df.columns)[:10]}...")
        print(f"Required features: {feature_names[:10]}...")
        
        # Check for missing features
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            print(f"WARNING: Missing features: {missing_features[:10]}...")
            print(f"Total missing features: {len(missing_features)}")
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
                print(f"Added missing feature '{feature}' with default value 0.0")
        
        # Select only the required features in the correct order
        X_pred_raw = df[feature_names].copy()
        print(f"Raw data shape: {X_pred_raw.shape}")
        
        # Apply preprocessing if available
        if imputer is not None and scaler is not None:
            print("Applying imputation and scaling...")
            # Impute missing values
            X_pred_imputed = pd.DataFrame(
                imputer.transform(X_pred_raw), 
                columns=X_pred_raw.columns, 
                index=X_pred_raw.index
            )
            # Scale features
            X_pred = pd.DataFrame(
                scaler.transform(X_pred_imputed), 
                columns=X_pred_imputed.columns, 
                index=X_pred_imputed.index
            )
            print("Preprocessing applied successfully")
        else:
            print("WARNING: No preprocessing objects available, using raw data")
            X_pred = X_pred_raw.fillna(0)
        
        print(f"Final prepared data shape: {X_pred.shape}")
        print(f"Data range - min: {X_pred.min().min():.4f}, max: {X_pred.max().max():.4f}")
        
        return X_pred
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available trained models"""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                continue
                
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                models.append({
                    "training_id": metadata["training_id"],
                    "mission": metadata["mission"],
                    "models": metadata["models"],
                    "metrics": metadata["metrics"],
                    "timestamp": metadata["timestamp"]
                })
            except:
                continue
        
        return models
    
    def _clean_csv_file(self, csv_file_path: str) -> str:
        """Clean and fix common CSV formatting issues"""
        try:
            # Read the file as text first
            with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                raise Exception("CSV file is empty")
            
            # Detect separator by analyzing the first few lines
            separators = [',', ';', '\t', '|']
            best_separator = ','
            max_fields = 0
            
            for sep in separators:
                sample_fields = len(lines[0].strip().split(sep))
                if sample_fields > max_fields:
                    max_fields = sample_fields
                    best_separator = sep
            
            print(f"Detected separator: '{best_separator}' with {max_fields} fields")
            
            # Get the expected number of fields from the header
            header = lines[0].strip()
            expected_fields = len(header.split(best_separator))
            print(f"Expected fields from header: {expected_fields}")
            
            # Clean lines that have inconsistent field counts
            cleaned_lines = [lines[0]]  # Keep header
            skipped_lines = 0
            
            for i, line in enumerate(lines[1:], 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Count fields in this line
                fields = line.split(best_separator)
                if len(fields) == expected_fields:
                    cleaned_lines.append(line)
                else:
                    print(f"Skipping line {i+1}: expected {expected_fields} fields, got {len(fields)}")
                    skipped_lines += 1
            
            print(f"Skipped {skipped_lines} lines with inconsistent field counts")
            
            # Write cleaned data to a new temporary file
            cleaned_file_path = csv_file_path.replace('.csv', '_cleaned.csv')
            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_lines))
            
            print(f"Created cleaned CSV file: {cleaned_file_path}")
            return cleaned_file_path
            
        except Exception as e:
            print(f"Error cleaning CSV file: {str(e)}")
            return csv_file_path  # Return original file if cleaning fails

    async def predict_batch(
        self, 
        mission: str, 
        dataset_filename: str, 
        model_name: str, 
        csv_file_path: str,
        true_labels_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make batch predictions from a CSV file"""
        try:
            print(f"Starting batch prediction: mission={mission}, dataset={dataset_filename}, model={model_name}")
            
            # Find the model
            model_path = await self._find_model(mission, dataset_filename, model_name)
            if not model_path:
                available_models = []
                for model_dir in self.models_dir.iterdir():
                    if model_dir.is_dir():
                        metadata_file = model_dir / "metadata.json"
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, "r") as f:
                                    metadata = json.load(f)
                                available_models.append(f"{metadata.get('mission', 'unknown')}-{metadata.get('models', [])}")
                            except:
                                continue
                
                raise ValueError(f"Model not found for mission={mission}, dataset={dataset_filename}, model={model_name}. Available models: {available_models}")
            
            print(f"Found model at: {model_path}")
            
            # Load model and metadata
            model, metadata, imputer, scaler = await self._load_model(model_path, model_name)
            print(f"Loaded model and metadata. Feature names: {metadata.get('feature_names', [])[:5]}...")
            print(f"Total features expected: {len(metadata.get('feature_names', []))}")
            print(f"Model type: {model_name}")
            print(f"Metadata keys: {list(metadata.keys())}")
            
            # Load CSV file with robust parsing
            cleaned_file_path = None
            try:
                # First, try standard CSV reading
                df = pd.read_csv(csv_file_path)
                print("Successfully loaded CSV with standard parsing")
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                print(f"Standard CSV parsing failed: {str(e)}")
                print("Trying different separators...")
                
                # Try different separators
                separators = [',', ';', '\t', '|']
                df = None
                for sep in separators:
                    try:
                        df = pd.read_csv(csv_file_path, sep=sep)
                        print(f"Successfully loaded CSV with separator '{sep}'")
                        break
                    except Exception as sep_error:
                        print(f"Separator '{sep}' failed: {str(sep_error)}")
                        continue
                
                if df is None:
                    print("All separator attempts failed. Attempting to clean CSV file...")
                    # Clean the CSV file
                    cleaned_file_path = self._clean_csv_file(csv_file_path)
                    
                    try:
                        # Try reading the cleaned file with detected separator
                        df = pd.read_csv(cleaned_file_path)
                        print("Successfully loaded cleaned CSV file")
                    except Exception as e2:
                        print(f"Cleaned CSV parsing failed: {str(e2)}")
                        # Try with different parsing options
                        try:
                            # Try with error handling for bad lines
                            df = pd.read_csv(cleaned_file_path, on_bad_lines='skip', encoding='utf-8')
                            print("Successfully loaded CSV by skipping bad lines")
                        except Exception as e3:
                            print(f"Skip bad lines failed: {str(e3)}")
                            try:
                                # Try with different separator detection
                                df = pd.read_csv(cleaned_file_path, sep=None, engine='python', on_bad_lines='skip')
                                print("Successfully loaded CSV with Python engine")
                            except Exception as e4:
                                print(f"Python engine failed: {str(e4)}")
                                try:
                                    # Try reading with low_memory=False
                                    df = pd.read_csv(cleaned_file_path, low_memory=False, on_bad_lines='skip')
                                    print("Successfully loaded CSV with low_memory=False")
                                except Exception as e5:
                                    print(f"All CSV parsing attempts failed: {str(e5)}")
                                    raise Exception(f"Unable to parse CSV file. Please check file format. Last error: {str(e5)}")
            
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"CSV columns: {list(df.columns)[:10]}...")
            print(f"Sample of raw CSV data (first 3 rows):")
            print(df.head(3))
            print(f"CSV data types: {df.dtypes}")
            print(f"CSV data range - min: {df.select_dtypes(include=[np.number]).min().min():.4f}, max: {df.select_dtypes(include=[np.number]).max().max():.4f}")
            
            # Extract true labels if column is specified
            true_labels = None
            if true_labels_column and true_labels_column in df.columns:
                true_labels = df[true_labels_column].values
                print(f"Extracted true labels from column '{true_labels_column}': {len(true_labels)} labels")
                # Remove the true labels column from features
                df = df.drop(columns=[true_labels_column])
            elif true_labels_column:
                print(f"Warning: True labels column '{true_labels_column}' not found in CSV")
            
            # Prepare input data
            X_pred = self._prepare_prediction_data(
                df.to_dict('records'), metadata["feature_names"], imputer, scaler
            )
            print(f"Prepared prediction data: {X_pred.shape}")
            print(f"Sample of prepared data (first 3 rows):")
            print(X_pred.head(3))
            print(f"Data types: {X_pred.dtypes}")
            print(f"Data range - min: {X_pred.min().min():.4f}, max: {X_pred.max().max():.4f}")
            
            # Make predictions
            if model_name == "xgboost":
                dmat = xgb.DMatrix(X_pred)
                predictions = model.predict(dmat)
            else:
                # For deep learning models, add channel dimension
                X_pred_seq = X_pred.values[..., None]
                predictions = model.predict(X_pred_seq, verbose=0).ravel()
            
            print(f"Generated {len(predictions)} predictions")
            print(f"Prediction range - min: {predictions.min():.4f}, max: {predictions.max():.4f}")
            print(f"Sample predictions (first 10): {predictions[:10]}")
            
            # Convert predictions to labels and confidence
            labels = ["CONFIRMED" if p > 0.5 else "FALSE POSITIVE" for p in predictions]
            confidence = [float(p) for p in predictions]
            
            result = {
                "predictions": [float(p) for p in predictions],
                "labels": labels,
                "confidence": confidence,
                "total_samples": len(predictions),
                "confirmed_count": sum(1 for label in labels if label == "CONFIRMED"),
                "false_positive_count": sum(1 for label in labels if label == "FALSE POSITIVE"),
                "average_confidence": float(np.mean(confidence))
            }
            
            # If true labels are provided, calculate comprehensive metrics
            if true_labels is not None:
                print("Calculating evaluation metrics...")
                y_true = np.array(true_labels)
                y_pred_binary = (np.array(predictions) > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self._calculate_comprehensive_metrics(y_true, y_pred_binary, predictions)
                result["metrics"] = metrics
                
                # Generate visualizations
                visualizations = await self._generate_visualizations(
                    y_true, predictions, X_pred, model, model_name, metadata
                )
                result["visualizations"] = visualizations
            
            print("Batch prediction completed successfully")
            
            # Clean up cleaned file if it was created
            if cleaned_file_path and os.path.exists(cleaned_file_path):
                try:
                    os.remove(cleaned_file_path)
                    print(f"Cleaned up temporary cleaned file: {cleaned_file_path}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up cleaned file: {cleanup_error}")
            
            return result
            
        except Exception as e:
            print(f"Error in predict_batch: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up cleaned file if it was created
            if 'cleaned_file_path' in locals() and cleaned_file_path and os.path.exists(cleaned_file_path):
                try:
                    os.remove(cleaned_file_path)
                    print(f"Cleaned up temporary cleaned file after error: {cleaned_file_path}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up cleaned file after error: {cleanup_error}")
            
            raise Exception(f"Batch prediction failed: {str(e)}")
