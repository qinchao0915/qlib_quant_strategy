#!/usr/bin/env python3
"""
V7 集成模型训练

包含：
- LGBM Conservative
- LGBM Bagging
- XGBoost
- 权重优化（SLSQP）
"""

import numpy as np
import pickle
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from pathlib import Path


class V7Trainer:
    """V7 集成训练器"""
    
    def __init__(self, pool="csi500"):
        """
        初始化
        
        Args:
            pool: 股票池，如 'csi500'
        """
        self.pool = pool
        self.models = {}
        self.weights = {}
        self.ic_scores = {}
    
    def train(self, X_train, y_train, X_valid, y_valid):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
            
        Returns:
            float: 集成模型的 IC
        """
        vp = {}  # 验证集预测
        
        # ========== 1. LGBM Conservative ==========
        print("Training LGBM Conservative...")
        lgb_cons = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=10,
            num_leaves=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_cons.fit(X_train, y_train)
        self.models['lgbm_conservative'] = lgb_cons
        vp['lgbm_conservative'] = lgb_cons.predict(X_valid)
        self.ic_scores['lgbm_conservative'] = self._calculate_ic(y_valid.values, vp['lgbm_conservative'])
        print(f"  LGBM Cons: {self.ic_scores['lgbm_conservative']:.4f}")
        
        # ========== 2. LGBM Bagging ==========
        print("Training LGBM Bagging...")
        lgb_bag = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=64,
            subsample=0.6,
            colsample_bytree=0.6,
            bagging_freq=5,
            bagging_fraction=0.6,
            random_state=43,
            n_jobs=-1,
            verbose=-1
        )
        lgb_bag.fit(X_train, y_train)
        self.models['lgbm_bagging'] = lgb_bag
        vp['lgbm_bagging'] = lgb_bag.predict(X_valid)
        self.ic_scores['lgbm_bagging'] = self._calculate_ic(y_valid.values, vp['lgbm_bagging'])
        print(f"  LGBM Bag: {self.ic_scores['lgbm_bagging']:.4f}")
        
        # ========== 3. XGBoost ==========
        print("Training XGBoost...")
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=44,
                n_jobs=-1,
                verbosity=0
            )
            xgb.fit(X_train, y_train)
            self.models['xgb'] = xgb
            vp['xgb'] = xgb.predict(X_valid)
            self.ic_scores['xgb'] = self._calculate_ic(y_valid.values, vp['xgb'])
            print(f"  XGB: {self.ic_scores['xgb']:.4f}")
        except ImportError:
            print("  ⚠️ XGBoost not installed, skipping")
            self.ic_scores['xgb'] = 0
        
        # ========== 4. 权重优化 ==========
        print("Optimizing ensemble weights...")
        mn = list(self.ic_scores.keys())
        
        # 初始权重（基于IC）
        ic_sum = sum(max(0, self.ic_scores[m]) for m in mn)
        if ic_sum > 0:
            iw = np.array([max(0, self.ic_scores[m]) / ic_sum for m in mn])
        else:
            iw = np.ones(len(mn)) / len(mn)
        
        # 目标函数：最大化集成IC
        def obj(w):
            ep = np.zeros(len(y_valid))
            for i, n in enumerate(mn):
                ep = ep + w[i] * vp[n]
            return -self._calculate_ic(y_valid.values, ep)
        
        # SLSQP优化
        r = minimize(
            obj,
            iw,
            method='SLSQP',
            bounds=[(0, 1)] * len(mn),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        self.weights = {n: r.x[i] for i, n in enumerate(mn)}
        
        # 计算集成IC
        ep = sum(self.weights[m] * vp[m] for m in mn)
        eic = self._calculate_ic(y_valid.values, ep)
        
        print(f"\n✅ Ensemble IC: {eic:.4f}")
        print(f"   Weights: {self.weights}")
        
        return eic
    
    @staticmethod
    def _calculate_ic(y_true, y_pred):
        """
        计算IC（信息系数）
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            float: IC值
        """
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        if m.sum() < 10:
            return 0
        return np.corrcoef(y_true[m], y_pred[m])[0, 1]
    
    def save(self, path, features, eic):
        """
        保存模型
        
        Args:
            path: 保存路径
            features: 特征列表
            eic: 集成IC
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'weights': self.weights,
                'features': features,
                'ic_scores': self.ic_scores,
                'ensemble_ic': eic,
                'type': f'v7_{self.pool}',
                'version': '7.0'
            }, f)
        
        print(f"✅ Model saved: {path}")
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.weights = data['weights']
        self.ic_scores = data['ic_scores']
        
        print(f"✅ Model loaded: {path}")
        print(f"   IC: {data['ensemble_ic']:.4f}")
        
        return data['features']
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            array: 预测值
        """
        predictions = []
        for name, model in self.models.items():
            if name in self.weights and self.weights[name] > 0:
                pred = model.predict(X)
                predictions.append(self.weights[name] * pred)
        
        return np.sum(predictions, axis=0)


if __name__ == '__main__':
    print("V7 Trainer ready")
    print("Usage:")
    print("  trainer = V7Trainer('csi500')")
    print("  ic = trainer.train(X_train, y_train, X_valid, y_valid)")
    print("  trainer.save('model.pkl', features, ic)")
