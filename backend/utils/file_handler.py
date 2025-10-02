import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiofiles
from fastapi import UploadFile

class FileHandler:
    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Create mission directories
        for mission in ["kepler", "k2", "tess"]:
            (self.datasets_dir / mission).mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile, mission: str) -> str:
        """Save uploaded file to appropriate mission directory"""
        mission_dir = self.datasets_dir / mission.lower()
        
        # Generate unique filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{mission.lower()}_{timestamp}_{file.filename}"
        file_path = mission_dir / filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return filename
    
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List all uploaded datasets"""
        datasets = []
        
        for mission_dir in self.datasets_dir.iterdir():
            if not mission_dir.is_dir():
                continue
                
            mission = mission_dir.name
            for file_path in mission_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.xlsx']:
                    # Get file info
                    stat = file_path.stat()
                    datasets.append({
                        "mission": mission,
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "created": pd.Timestamp.fromtimestamp(stat.st_ctime).isoformat(),
                        "path": str(file_path)
                    })
        
        return datasets
    
    def get_dataset_path(self, mission: str, filename: str) -> Path:
        """Get path to a specific dataset"""
        return self.datasets_dir / mission.lower() / filename
    
    def dataset_exists(self, mission: str, filename: str) -> bool:
        """Check if dataset exists"""
        return self.get_dataset_path(mission, filename).exists()
