#!/usr/bin/env python3
"""
特征工程工具

功能：
- 技术指标计算
- Alpha因子生成
- 特征标准化
"""

import pandas as pd
import numpy as np


class FeatureUtils:
    """特征工程工具类"""
    
    @staticmethod
    def calculate_ma(df, window=5, field='$close'):
        """计算移动平均线"""
        return df[field].rolling(window=window).mean()
    
    @staticmethod
    def calculate_rsi(df, window=14, field='$close'):
        """计算RSI指标"""
        delta = df[field].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9, field='$close'):
        """计算MACD指标"""
        ema_fast = df[field].ewm(span=fast).mean()
        ema_slow = df[field].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def calculate_bollinger(df, window=20, num_std=2, field='$close'):
        """计算布林带"""
        ma = df[field].rolling(window=window).mean()
        std = df[field].rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, ma, lower
    
    @staticmethod
    def add_technical_indicators(df):
        """添加常用技术指标"""
        # 移动平均线
        df['MA5'] = FeatureUtils.calculate_ma(df, 5)
        df['MA10'] = FeatureUtils.calculate_ma(df, 10)
        df['MA20'] = FeatureUtils.calculate_ma(df, 20)
        
        # RSI
        df['RSI'] = FeatureUtils.calculate_rsi(df)
        
        # MACD
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = FeatureUtils.calculate_macd(df)
        
        # 布林带
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = FeatureUtils.calculate_bollinger(df)
        
        return df


if __name__ == '__main__':
    # 测试
    import numpy as np
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        '$close': np.random.randn(100).cumsum() + 100,
        '$open': np.random.randn(100).cumsum() + 100,
        '$high': np.random.randn(100).cumsum() + 102,
        '$low': np.random.randn(100).cumsum() + 98,
        '$volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    df = FeatureUtils.add_technical_indicators(df)
    print(df.tail())
