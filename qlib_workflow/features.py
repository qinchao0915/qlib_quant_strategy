#!/usr/bin/env python3
"""
Qlib Features 组件

功能：
- Alpha 因子计算
- 特征预处理
- 特征选择
"""

import pandas as pd
import numpy as np


class QlibFeatures:
    """Qlib 风格特征工程"""
    
    def __init__(self, config=None):
        """
        初始化
        
        Args:
            config: 特征配置
        """
        self.config = config or {}
        self.feature_names = []
    
    def calc_alpha_features(self, df):
        """
        计算 Alpha 因子
        
        类似 Qlib 的 Alpha360/158
        """
        features = []
        
        # 按股票分组计算
        for symbol, group in df.groupby(level='symbol'):
            group = group.sort_index()
            
            # 价格特征
            group['$close'] = group['close']
            group['$open'] = group['open']
            group['$high'] = group['high']
            group['$low'] = group['low']
            group['$volume'] = group['volume']
            
            # 收益率
            group['$returns'] = group['$close'].pct_change()
            
            # 移动平均
            for window in [5, 10, 20, 30, 60]:
                group[f'$ma{window}'] = group['$close'].rolling(window).mean()
                group[f'$returns_{window}d'] = group['$close'].pct_change(window)
            
            # 波动率
            group['$volatility_20d'] = group['$returns'].rolling(20).std()
            
            # RSI
            delta = group['$close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            group['$rsi_14'] = 100 - (100 / (1 + rs))
            
            features.append(group)
        
        result = pd.concat(features)
        
        # 记录特征名
        self.feature_names = [c for c in result.columns if c.startswith('$')]
        
        return result
    
    def preprocess(self, df, method='standard'):
        """
        特征预处理
        
        Args:
            df: 特征数据
            method: 预处理方法
        """
        if method == 'standard':
            # 标准化
            for col in self.feature_names:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = (df[col] - mean) / (std + 1e-8)
        
        elif method == 'rank':
            # 截面排名（类似 Qlib）
            for col in self.feature_names:
                if col in df.columns:
                    df[col] = df.groupby(level='date')[col].rank(pct=True)
        
        elif method == 'zscore':
            # 截面 Z-Score
            for col in self.feature_names:
                if col in df.columns:
                    mean = df.groupby(level='date')[col].transform('mean')
                    std = df.groupby(level='date')[col].transform('std')
                    df[col] = (df[col] - mean) / (std + 1e-8)
        
        return df
    
    def get_feature_names(self):
        """获取特征名列表"""
        return self.feature_names
