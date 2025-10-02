import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self):
        pass
    
    async def process_dataset(self, filename: str, mission: str) -> Dict[str, Any]:
        """Process and validate dataset"""
        file_path = Path(f"datasets/{mission.lower()}/{filename}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            # Read dataset
            df = pd.read_csv(file_path, comment="#", sep=",", engine="python")
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Check for required columns based on mission
            required_columns = self._get_required_columns(mission)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} for {mission} mission")
            
            return {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "missing_columns": missing_columns,
                "data_types": df.dtypes.to_dict()
            }
            
        except Exception as e:
            raise Exception(f"Error processing dataset: {str(e)}")
    
    def _get_required_columns(self, mission: str) -> List[str]:
        """Get required columns for each mission"""
        if mission.lower() == "kepler":
            return [
                "koi_disposition",
                "koi_period",
                "koi_duration", 
                "koi_depth",
                "koi_prad"
            ]
        elif mission.lower() == "k2":
            return [
                "disposition",  # or koi_disposition
                "pl_orbper",   # or koi_period
                "pl_trandur",  # or koi_duration
                "pl_trandep",  # or koi_depth
                "pl_rade"      # or koi_prad
            ]
        elif mission.lower() == "tess":
            return [
                "tfopwg_disp", # or koi_disposition
                "toi_period",  # or koi_period
                "toi_duration", # or koi_duration
                "toi_depth",   # or koi_depth
                "pl_rade"      # or koi_prad
            ]
        else:
            return []
    
    def validate_prediction_data(self, data: List[Dict[str, Any]], feature_names: List[str]) -> bool:
        """Validate prediction input data"""
        if not data:
            return False
        
        # Check if all required features are present
        for feature in feature_names:
            if not any(feature in row for row in data):
                return False
        
        return True
    
    def get_sample_data(self, mission: str, filename: str, n_samples: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from dataset for testing predictions"""
        file_path = Path(f"datasets/{mission.lower()}/{filename}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path, comment="#", sep=",", engine="python")
        
        # Get sample rows
        sample_df = df.head(n_samples)
        
        # Convert to list of dictionaries
        return sample_df.to_dict('records')
