import os
import json
import uuid
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention
import shap

from utils.data_processor import DataProcessor

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class TrainingManager:
    def __init__(self):
        self.training_jobs = {}
        self.models_dir = Path("models/trained")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_processor = DataProcessor()
        
    async def start_training(
        self, 
        mission: str, 
        dataset_filename: str, 
        models: List[str],
        test_size: float = 0.2,
        background_tasks=None
    ) -> str:
        """Start training process"""
        training_id = str(uuid.uuid4())
        
        # Initialize training job
        self.training_jobs[training_id] = {
            "training_id": training_id,
            "mission": mission,
            "models": models,
            "status": "started",
            "progress": 0.0,
            "start_time": datetime.now().isoformat(),
            "metrics": None,
            "error": None
        }
        
        # Start training in background
        if background_tasks:
            background_tasks.add_task(
                self._train_models, 
                training_id, 
                mission, 
                dataset_filename, 
                models, 
                test_size
            )
        
        return training_id
    
    async def _train_models(
        self, 
        training_id: str, 
        mission: str, 
        dataset_filename: str, 
        models: List[str],
        test_size: float
    ):
        """Train models in background"""
        try:
            # Update status
            self.training_jobs[training_id]["status"] = "loading_data"
            self.training_jobs[training_id]["progress"] = 10.0
            
            # Load and process data
            df, X_tab, y, imputer, scaler = await self._load_and_process_data(mission, dataset_filename)
            
            # Split data
            X_train_tab, X_test_tab, y_train, y_test = train_test_split(
                X_tab, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
            )
            
            # Update status
            self.training_jobs[training_id]["status"] = "training"
            self.training_jobs[training_id]["progress"] = 20.0
            
            # Train models
            trained_models = {}
            model_metrics = {}
            
            for i, model_name in enumerate(models):
                self.training_jobs[training_id]["progress"] = 20.0 + (i / len(models)) * 60.0
                
                if model_name == "xgboost":
                    model, metrics = await self._train_xgboost(X_train_tab, X_test_tab, y_train, y_test)
                elif model_name == "cnn1d":
                    model, metrics = await self._train_cnn1d(X_train_tab, X_test_tab, y_train, y_test)
                elif model_name == "resnet1d":
                    model, metrics = await self._train_resnet1d(X_train_tab, X_test_tab, y_train, y_test)
                elif model_name == "transformer":
                    model, metrics = await self._train_transformer(X_train_tab, X_test_tab, y_train, y_test)
                else:
                    continue
                
                trained_models[model_name] = model
                model_metrics[model_name] = metrics
            
            # Update status
            self.training_jobs[training_id]["status"] = "saving"
            self.training_jobs[training_id]["progress"] = 80.0
            
            # Save models
            await self._save_models(training_id, mission, trained_models, model_metrics, X_test_tab.columns.tolist(), imputer, scaler)
            
            # Update final status
            self.training_jobs[training_id]["status"] = "completed"
            self.training_jobs[training_id]["progress"] = 100.0
            self.training_jobs[training_id]["metrics"] = model_metrics
            self.training_jobs[training_id]["end_time"] = datetime.now().isoformat()
            
        except Exception as e:
            self.training_jobs[training_id]["status"] = "failed"
            self.training_jobs[training_id]["error"] = str(e)
            self.training_jobs[training_id]["end_time"] = datetime.now().isoformat()
    
    async def _load_and_process_data(self, mission: str, dataset_filename: str):
        """Load and process dataset based on mission"""
        file_path = Path(f"datasets/{mission}/{dataset_filename}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path, comment="#", sep=",", engine="python")
        
        if mission.lower() == "kepler":
            return self._process_kepler_data(df)
        elif mission.lower() == "k2":
            return self._process_k2_data(df)
        elif mission.lower() == "tess":
            return self._process_tess_data(df)
        else:
            raise ValueError(f"Unsupported mission: {mission}")
    
    def _process_kepler_data(self, df: pd.DataFrame):
        """Process Kepler KOI dataset"""
        # Keep only CONFIRMED vs FALSE POSITIVE
        df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
        df["label"] = df["koi_disposition"].map({"CONFIRMED": 1, "FALSE POSITIVE": 0})
        
        # Feature selection
        candidate_features = [
            "koi_period", "koi_duration", "koi_depth", "koi_prad",
            "koi_score",
            "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
            "koi_steff", "koi_slogg", "koi_srad",
            "koi_time0bk", "koi_kepmag"
        ]
        features = [f for f in candidate_features if f in df.columns]
        
        X_tab = df[features].fillna(0)
        y = df["label"].astype(int).values
        
        return df, X_tab, y
    
    def _process_k2_data(self, df: pd.DataFrame):
        """Process K2 dataset"""
        # Find label column
        label_col = None
        for cand in ["koi_disposition", "disposition"]:
            if cand in df.columns:
                label_col = cand
                break
        
        if label_col is None:
            raise ValueError("No disposition column found")
        
        # Normalize labels
        valid_map = {"CONFIRMED": 1, "FALSE POSITIVE": 0}
        df = df[df[label_col].isin(valid_map.keys())].copy()
        df["label"] = df[label_col].map(valid_map).astype(int)
        
        # Feature mapping
        feature_candidates = {
            "koi_period": ["koi_period", "pl_orbper"],
            "koi_duration": ["koi_duration", "pl_trandur"],
            "koi_depth": ["koi_depth", "pl_trandep"],
            "koi_prad": ["koi_prad", "pl_rade"],
            "koi_score": ["koi_score"],
            "koi_fpflag_nt": ["koi_fpflag_nt"],
            "koi_fpflag_ss": ["koi_fpflag_ss"],
            "koi_fpflag_co": ["koi_fpflag_co"],
            "koi_fpflag_ec": ["koi_fpflag_ec"],
            "koi_steff": ["koi_steff", "st_teff"],
            "koi_slogg": ["koi_slogg", "st_logg"],
            "koi_srad": ["koi_srad", "st_rad"],
            "koi_time0bk": ["koi_time0bk", "pl_tranmid"],
            "koi_kepmag": ["koi_kepmag", "sy_kepmag", "kic_kepmag", "kepmag"]
        }
        
        # Build feature mapping
        selected_mapping = {}
        for logical, candidates in feature_candidates.items():
            for c in candidates:
                if c in df.columns:
                    selected_mapping[logical] = c
                    break
        
        if len(selected_mapping) >= 6:
            X_tab_raw = df[[selected_mapping[k] for k in selected_mapping.keys()]].copy()
            X_tab_raw.columns = list(selected_mapping.keys())
        else:
            # Fallback to numeric columns
            exclude_like = set([label_col, "label", "pl_name", "hostname", "kepoi_name", "kepid"])
            numeric_cols = [c for c in df.columns if c not in exclude_like and pd.api.types.is_numeric_dtype(df[c])]
            X_tab_raw = df[numeric_cols].copy()
        
        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_tab_imputed = pd.DataFrame(imputer.fit_transform(X_tab_raw), columns=X_tab_raw.columns, index=X_tab_raw.index)
        
        # Scale features
        scaler = StandardScaler()
        X_tab = pd.DataFrame(scaler.fit_transform(X_tab_imputed), columns=X_tab_imputed.columns, index=X_tab_imputed.index)
        y = df["label"].values
        
        return df, X_tab, y, imputer, scaler
    
    def _process_tess_data(self, df: pd.DataFrame):
        """Process TESS dataset"""
        # Determine if KOI or TOI
        is_koi = "koi_disposition" in df.columns
        is_toi = "tfopwg_disp" in df.columns or "toi" in df.columns
        
        if is_koi:
            df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
            df["label"] = df["koi_disposition"].map({"CONFIRMED": 1, "FALSE POSITIVE": 0}).astype(int)
        elif is_toi:
            disp_col = "tfopwg_disp"
            pos_vals = set(v for v in ["PC", "CP", "KP", "CONFIRMED"] if v in df[disp_col].unique())
            neg_vals = set(v for v in ["FP", "FALSE POSITIVE"] if v in df[disp_col].unique())
            df = df[df[disp_col].isin(pos_vals.union(neg_vals))].copy()
            df["label"] = df[disp_col].apply(lambda v: 1 if v in pos_vals else 0).astype(int)
        else:
            raise ValueError("Unrecognized schema")
        
        # Feature selection
        def pick_first(cols, candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None
        
        cols = set(df.columns)
        feature_candidates = {
            "period": ["koi_period", "toi_period", "pl_orbper"],
            "duration": ["koi_duration", "toi_duration", "pl_trandur"],
            "depth": ["koi_depth", "toi_depth", "pl_trandep"],
            "prad": ["koi_prad", "pl_rade", "toi_prad"],
            "score": ["koi_score", "toi_score"],
            "fpflag_nt": ["koi_fpflag_nt"],
            "fpflag_ss": ["koi_fpflag_ss"],
            "fpflag_co": ["koi_fpflag_co"],
            "fpflag_ec": ["koi_fpflag_ec"],
            "teff": ["koi_steff", "st_teff"],
            "logg": ["koi_slogg", "st_logg"],
            "srad": ["koi_srad", "st_rad"],
            "t0": ["koi_time0bk", "toi_t0", "pl_tranmid"],
            "mag": ["koi_kepmag", "st_vj", "st_gaiamag"]
        }
        
        selected_feature_cols = []
        for _, cands in feature_candidates.items():
            col = pick_first(cols, cands)
            if col:
                selected_feature_cols.append(col)
        
        X_tab_raw = df[selected_feature_cols].copy()
        
        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_tab_imputed = pd.DataFrame(imputer.fit_transform(X_tab_raw), columns=X_tab_raw.columns, index=X_tab_raw.index)
        
        # Scale features
        scaler = StandardScaler()
        X_tab = pd.DataFrame(scaler.fit_transform(X_tab_imputed), columns=X_tab_imputed.columns, index=X_tab_imputed.index)
        y = df["label"].astype(int).values
        
        return df, X_tab, y, imputer, scaler
    
    async def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": RANDOM_SEED
        }
        
        model = xgb.train(xgb_params, dtrain, num_boost_round=400)
        y_prob = model.predict(dtest)
        
        metrics = self._calculate_metrics(y_test, y_prob)
        return model, metrics
    
    async def _train_cnn1d(self, X_train, X_test, y_train, y_test):
        """Train 1D CNN model"""
        # For CNN, we need to reshape the data to have a sequence dimension
        # We'll use the features as a 1D sequence
        X_train_seq = X_train.values[..., None]  # Add channel dimension
        X_test_seq = X_test.values[..., None]
        
        model = self._build_cnn1d(X_train_seq.shape[1:])
        
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=8,
            batch_size=256,
            verbose=0
        )
        
        y_prob = model.predict(X_test_seq, verbose=0).ravel()
        metrics = self._calculate_metrics(y_test, y_prob)
        
        return model, metrics
    
    async def _train_resnet1d(self, X_train, X_test, y_train, y_test):
        """Train 1D ResNet model"""
        X_train_seq = X_train.values[..., None]
        X_test_seq = X_test.values[..., None]
        
        model = self._build_resnet1d(X_train_seq.shape[1:])
        
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=8,
            batch_size=256,
            verbose=0
        )
        
        y_prob = model.predict(X_test_seq, verbose=0).ravel()
        metrics = self._calculate_metrics(y_test, y_prob)
        
        return model, metrics
    
    async def _train_transformer(self, X_train, X_test, y_train, y_test):
        """Train Transformer model"""
        X_train_seq = X_train.values[..., None]
        X_test_seq = X_test.values[..., None]
        
        model = self._build_transformer1d(X_train_seq.shape[1:])
        
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=8,
            batch_size=256,
            verbose=0
        )
        
        y_prob = model.predict(X_test_seq, verbose=0).ravel()
        metrics = self._calculate_metrics(y_test, y_prob)
        
        return model, metrics
    
    def _build_cnn1d(self, input_shape):
        """Build 1D CNN model"""
        model = models.Sequential([
            layers.Conv1D(32, 7, padding="same", activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, padding="same", activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    
    def _build_resnet1d(self, input_shape):
        """Build 1D ResNet model"""
        def residual_block(x, filters, k=3):
            shortcut = x
            x = layers.Conv1D(filters, k, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv1D(filters, k, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([shortcut, x])
            x = layers.ReLU()(x)
            return x
        
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 7, padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = residual_block(x, 64)
        x = layers.MaxPooling1D(2)(x)
        x = residual_block(x, 64)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    
    def _build_transformer1d(self, input_shape, num_heads=4, key_dim=32):
        """Build 1D Transformer model"""
        inp = layers.Input(shape=input_shape)
        x = layers.LayerNormalization()(inp)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = layers.Add()([inp, x])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, 1, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inp, out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    
    def _calculate_metrics(self, y_true, y_prob):
        """Calculate evaluation metrics"""
        y_pred = (y_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        spec = recall_score(y_true, y_pred, pos_label=0)
        f1 = f1_score(y_true, y_pred)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
    
    async def _save_models(self, training_id: str, mission: str, models: Dict, metrics: Dict, feature_names: List[str], imputer, scaler):
        """Save trained models"""
        model_dir = self.models_dir / training_id
        model_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "training_id": training_id,
            "mission": mission,
            "models": list(models.keys()),
            "metrics": metrics,
            "feature_names": feature_names,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save preprocessing objects
        joblib.dump(imputer, model_dir / "imputer.pkl")
        joblib.dump(scaler, model_dir / "scaler.pkl")
        
        # Save models
        for model_name, model in models.items():
            if model_name == "xgboost":
                model.save_model(str(model_dir / f"{model_name}.json"))
            else:
                model.save(str(model_dir / f"{model_name}.h5"))
    
    async def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get training status"""
        if training_id not in self.training_jobs:
            raise ValueError(f"Training job {training_id} not found")
        
        return self.training_jobs[training_id]
    
    async def get_training_history(self) -> List[Dict[str, Any]]:
        """Get all training history"""
        return list(self.training_jobs.values())
