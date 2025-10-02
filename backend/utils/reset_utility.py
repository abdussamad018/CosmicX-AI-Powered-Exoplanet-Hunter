"""
Reset utility for clearing all training and prediction data
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any


class ResetUtility:
    """Utility class for resetting training and prediction data"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models" / "trained"
        self.datasets_dir = self.base_dir / "datasets"
        
    def reset_all_data(self) -> Dict[str, Any]:
        """Reset all training and prediction data"""
        results = {
            "models_removed": 0,
            "datasets_removed": 0,
            "temp_files_removed": 0,
            "errors": []
        }
        
        try:
            # Reset trained models
            models_result = self.reset_trained_models()
            results["models_removed"] = models_result["removed"]
            results["errors"].extend(models_result["errors"])
            
            # Reset datasets
            datasets_result = self.reset_datasets()
            results["datasets_removed"] = datasets_result["removed"]
            results["errors"].extend(datasets_result["errors"])
            
            # Clean temporary files
            temp_result = self.clean_temp_files()
            results["temp_files_removed"] = temp_result["removed"]
            results["errors"].extend(temp_result["errors"])
            
        except Exception as e:
            results["errors"].append(f"Unexpected error during reset: {str(e)}")
        
        return results
    
    def reset_trained_models(self) -> Dict[str, Any]:
        """Remove all trained model directories and files"""
        result = {"removed": 0, "errors": []}
        
        try:
            if self.models_dir.exists():
                # Count directories before removal
                model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
                result["removed"] = len(model_dirs)
                
                # Remove all model directories
                for model_dir in model_dirs:
                    try:
                        shutil.rmtree(model_dir)
                        print(f"Removed model directory: {model_dir}")
                    except Exception as e:
                        error_msg = f"Error removing {model_dir}: {str(e)}"
                        result["errors"].append(error_msg)
                        print(error_msg)
                
                print(f"Removed {result['removed']} trained model directories")
            else:
                print("No trained models directory found")
                
        except Exception as e:
            error_msg = f"Error resetting trained models: {str(e)}"
            result["errors"].append(error_msg)
            print(error_msg)
        
        return result
    
    def reset_datasets(self) -> Dict[str, Any]:
        """Remove all uploaded datasets"""
        result = {"removed": 0, "errors": []}
        
        try:
            if self.datasets_dir.exists():
                # Count files before removal
                dataset_files = []
                for mission_dir in self.datasets_dir.iterdir():
                    if mission_dir.is_dir():
                        for file in mission_dir.iterdir():
                            if file.is_file() and file.suffix == '.csv':
                                dataset_files.append(file)
                
                result["removed"] = len(dataset_files)
                
                # Remove all dataset files
                for dataset_file in dataset_files:
                    try:
                        dataset_file.unlink()
                        print(f"Removed dataset: {dataset_file}")
                    except Exception as e:
                        error_msg = f"Error removing {dataset_file}: {str(e)}"
                        result["errors"].append(error_msg)
                        print(error_msg)
                
                print(f"Removed {result['removed']} dataset files")
            else:
                print("No datasets directory found")
                
        except Exception as e:
            error_msg = f"Error resetting datasets: {str(e)}"
            result["errors"].append(error_msg)
            print(error_msg)
        
        return result
    
    def clean_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        result = {"removed": 0, "errors": []}
        
        try:
            # Common temporary file patterns
            temp_patterns = [
                "temp_*.csv",
                "temp_*.json",
                "*.tmp",
                "*.temp",
                "__pycache__",
                "*.pyc"
            ]
            
            removed_files = []
            
            # Clean temp files in base directory
            for pattern in temp_patterns:
                for file_path in glob.glob(str(self.base_dir / pattern)):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            removed_files.append(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            removed_files.append(file_path)
                    except Exception as e:
                        error_msg = f"Error removing {file_path}: {str(e)}"
                        result["errors"].append(error_msg)
                        print(error_msg)
            
            # Clean __pycache__ directories recursively
            for root, dirs, files in os.walk(self.base_dir):
                for dir_name in dirs[:]:  # Use slice to avoid modifying list while iterating
                    if dir_name == "__pycache__":
                        pycache_path = os.path.join(root, dir_name)
                        try:
                            shutil.rmtree(pycache_path)
                            removed_files.append(pycache_path)
                        except Exception as e:
                            error_msg = f"Error removing {pycache_path}: {str(e)}"
                            result["errors"].append(error_msg)
                            print(error_msg)
                        dirs.remove(dir_name)  # Don't recurse into removed directory
            
            result["removed"] = len(removed_files)
            print(f"Removed {result['removed']} temporary files and directories")
            
        except Exception as e:
            error_msg = f"Error cleaning temp files: {str(e)}"
            result["errors"].append(error_msg)
            print(error_msg)
        
        return result
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of current data"""
        summary = {
            "trained_models": 0,
            "datasets": 0,
            "total_size_mb": 0
        }
        
        try:
            # Count trained models
            if self.models_dir.exists():
                model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
                summary["trained_models"] = len(model_dirs)
            
            # Count datasets and calculate size
            if self.datasets_dir.exists():
                dataset_files = []
                for mission_dir in self.datasets_dir.iterdir():
                    if mission_dir.is_dir():
                        for file in mission_dir.iterdir():
                            if file.is_file() and file.suffix == '.csv':
                                dataset_files.append(file)
                
                summary["datasets"] = len(dataset_files)
                
                # Calculate total size
                total_size = 0
                for file in dataset_files:
                    total_size += file.stat().st_size
                
                summary["total_size_mb"] = round(total_size / (1024 * 1024), 2)
                
        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
        
        return summary


# Convenience function for easy import
def reset_all_data(base_dir: str = ".") -> Dict[str, Any]:
    """Convenience function to reset all data"""
    reset_util = ResetUtility(base_dir)
    return reset_util.reset_all_data()


def get_data_summary(base_dir: str = ".") -> Dict[str, Any]:
    """Convenience function to get data summary"""
    reset_util = ResetUtility(base_dir)
    return reset_util.get_data_summary()


if __name__ == "__main__":
    # Test the reset utility
    print("=== Data Reset Utility ===")
    
    # Show current data summary
    print("\nCurrent data summary:")
    summary = get_data_summary()
    print(f"Trained models: {summary['trained_models']}")
    print(f"Datasets: {summary['datasets']}")
    print(f"Total size: {summary['total_size_mb']} MB")
    
    # Ask for confirmation
    response = input("\nDo you want to reset all data? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        print("\nResetting all data...")
        results = reset_all_data()
        
        print(f"\nReset completed:")
        print(f"Models removed: {results['models_removed']}")
        print(f"Datasets removed: {results['datasets_removed']}")
        print(f"Temp files removed: {results['temp_files_removed']}")
        
        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
    else:
        print("Reset cancelled.")
