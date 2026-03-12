#!/usr/bin/env python3
"""
V7 特征工程

生成 53 个特征：
- 动量 8个
- 波动率 6个
- 均线 8个
- 布林带 3个
- RSI 4个
- MACD 5个
- 成交量 8个
- 振幅 5个
- 资金流 5个
- 市值 3个
"""

import pandas as pd
import numpy as np


class V7FeatureEngineer:
    """V7 特征工程器"""
    
    @staticmethod
    def calculate_all_features(price_df):
        """
        计算所有特征
        
        Args:
            price_df: 价格数据 DataFrame
            
        Returns:
            DataFrame: 添加特征后的数据
        """
        df = price_df.copy()
        
        # 按股票分组计算
        features_list = []
        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('date')
            group = V7FeatureEngineer._calculate_features_for_stock(group)
            features_list.append(group)
        
        return pd.concat(features_list, ignore_index=True)
    
    @staticmethod
    def _calculate_features_for_stock(df):
        """为单只股票计算特征"""
        
        # ========== 1. 动量特征 (8个) ==========
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        df['return_60d'] = df['close'].pct_change(60)
        df['return_accel'] = df['return_5d'] - df['return_10d']
        df['return_volatility'] = df['return_1d'].rolling(20).std()
        
        # ========== 2. 波动率特征 (6个) ==========
        df['volatility_5d'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        df['volatility_10d'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        df['volatility_20d'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volatility_60d'] = df['close'].rolling(60).std() / df['close'].rolling(60).mean()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        df['amplitude_20d'] = (df['high'] - df['low']).rolling(20).mean() / df['close']
        
        # ========== 3. 均线特征 (8个) ==========
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['price_to_ma5'] = df['close'] / df['ma5']
        df['price_to_ma10'] = df['close'] / df['ma10']
        df['price_to_ma20'] = df['close'] / df['ma20']
        df['price_to_ma60'] = df['close'] / df['ma60']
        df['ma5_to_ma20'] = df['ma5'] / df['ma20']
        
        # ========== 4. 布林带特征 (3个) ==========
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_upper_ratio'] = df['close'] / df['bb_upper']
        
        # ========== 5. RSI 特征 (4个) ==========
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        df['rsi_6'] = 100 - (100 / (1 + rs))
        
        gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
        rs = gain / loss
        df['rsi_12'] = 100 - (100 / (1 + rs))
        
        gain = (delta.where(delta > 0, 0)).rolling(window=24).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=24).mean()
        rs = gain / loss
        df['rsi_24'] = 100 - (100 / (1 + rs))
        
        df['rsi_diff'] = df['rsi_6'] - df['rsi_24']
        
        # ========== 6. MACD 特征 (5个) ==========
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_trend'] = df['macd'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # ========== 7. 成交量特征 (8个) ==========
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['turnover_ratio'] = df['volume'] / df['volume'].rolling(60).mean()
        df['volume_price_trend'] = df['volume'] * df['return_1d']
        df['volume_spike'] = (df['volume'] > 2 * df['volume_ma20']).astype(int)
        
        # 资金流（简化计算）
        df['money_flow'] = df['volume'] * (df['close'] + df['high'] + df['low']) / 3
        df['net_money_flow'] = df['money_flow'] * np.sign(df['close'] - df['open'])
        
        # ========== 8. 振幅特征 (5个) ==========
        df['amplitude_5d'] = ((df['high'] - df['low']) / df['low']).rolling(5).mean()
        df['amplitude_10d'] = ((df['high'] - df['low']) / df['low']).rolling(10).mean()
        df['amplitude_60d'] = ((df['high'] - df['low']) / df['low']).rolling(60).mean()
        df['amplitude_ratio'] = df['amplitude_5d'] / df['amplitude_20d']
        
        # ========== 9. 资金流特征 (5个) ==========
        df['big_money_ratio'] = df['volume'].rolling(5).apply(lambda x: (x > x.quantile(0.8)).mean())
        df['small_money_ratio'] = df['volume'].rolling(5).apply(lambda x: (x < x.quantile(0.2)).mean())
        df['money_flow_ma5'] = df['money_flow'].rolling(5).mean()
        df['money_flow_ma20'] = df['money_flow'].rolling(20).mean()
        df['money_flow_trend'] = df['money_flow_ma5'] / df['money_flow_ma20']
        
        # ========== 10. 市值特征 (3个) ==========
        # 市值需要额外数据，这里用价格作为代理
        df['market_cap_proxy'] = df['close'] * df['volume']
        df['market_cap_log'] = np.log(df['market_cap_proxy'] + 1)
        
        # ========== 目标变量 ==========
        df['label'] = df['close'].shift(-5) / df['close'] - 1  # 未来5日收益
        
        return df
    
    @staticmethod
    def get_feature_cols():
        """获取特征列名列表"""
        return [
            # 动量
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d', 'return_60d',
            'return_accel', 'return_volatility',
            # 波动率
            'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
            'volatility_ratio', 'amplitude_20d',
            # 均线
            'ma5', 'ma10', 'ma20', 'ma60', 'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
            'price_to_ma60', 'ma5_to_ma20',
            # 布林带
            'bb_position', 'bb_width', 'bb_upper_ratio',
            # RSI
            'rsi_6', 'rsi_12', 'rsi_24', 'rsi_diff',
            # MACD
            'macd', 'macd_signal', 'macd_hist', 'macd_cross', 'macd_trend',
            # 成交量
            'volume_ma5', 'volume_ma20', 'volume_ratio', 'turnover_ratio',
            'volume_price_trend', 'volume_spike', 'money_flow', 'net_money_flow',
            # 振幅
            'amplitude_5d', 'amplitude_10d', 'amplitude_60d', 'amplitude_ratio',
            # 资金流
            'big_money_ratio', 'small_money_ratio', 'money_flow_ma5', 'money_flow_ma20',
            'money_flow_trend',
            # 市值
            'market_cap_proxy', 'market_cap_log'
        ]


if __name__ == '__main__':
    # 测试
    print("特征数量:", len(V7FeatureEngineer.get_feature_cols()))
    print("特征列表:", V7FeatureEngineer.get_feature_cols()[:10], "...")
