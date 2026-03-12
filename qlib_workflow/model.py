#!/usr/bin/env python3
"""
Qlib Model 组件

功能：
- 模型训练
- 模型预测
- 模型管理
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from lightgbm import LGBMRegressor


class QlibModel:
    """Qlib 风格模型"""
    
    def __init__(self, model_type='lightgbm', model_params=None):
        """
        初始化
        
        Args:
            model_type: 模型类型
            model_params: 模型参数
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_importance = None
    
    def fit(self, dataset):
        """
        训练模型
        
        Args:
            dataset: 数据集 (X_train, y_train, X_valid, y_valid)
        """
        X_train, y_train, X_valid, y_valid = dataset
        
        if self.model_type == 'lightgbm':
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 64,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                **self.model_params
            }
            
            self.model = LGBMRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[]
            )
            
            # 特征重要性
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        print(f"✅ 模型训练完成: {self.model_type}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            array: 预测值
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        评估模型
        
        计算 IC
        """
        pred = self.predict(X)
        
        # 计算 IC
        mask = ~(np.isnan(y) | np.isnan(pred))
        if mask.sum() < 10:
            return 0
        
        ic = np.corrcoef(y[mask], pred[mask])[0, 1]
        return ic
    
    def save(self, path):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_importance': self.feature_importance
            }, f)
        
        print(f"💾 模型已保存: {path}")
    
    def load(self, path):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_importance = data['feature_importance']
        
        print(f"📂 模型已加载: {path}")
