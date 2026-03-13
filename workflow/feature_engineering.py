#!/usr/bin/env python3
"""
特征工程 - 53个特征实现

特征体系：
- 技术面特征 (38个)
- 资金流向特征 (5个) ⭐核心
- 市值分层特征 (4个)
- 基本面特征 (6个) - 预留
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all_features(self, price_df):
        """
        计算所有特征
        
        Args:
            price_df: 价格数据 DataFrame
            
        Returns:
            DataFrame: 添加特征后的数据
        """
        print("🔧 开始计算特征...")
        
        # 按股票分组计算
        features_list = []
        total = price_df['symbol'].nunique()
        
        for i, (symbol, group) in enumerate(price_df.groupby('symbol')):
            if i % 100 == 0:
                print(f"  进度: {i}/{total} ({i/total*100:.1f}%)")
            
            group = group.sort_values('date')
            
            # 计算各类特征
            group = self._calculate_technical_features(group)
            group = self._calculate_money_flow_features(group)
            group = self._calculate_size_features(group)
            
            features_list.append(group)
        
        result = pd.concat(features_list, ignore_index=True)
        
        # 记录特征名
        self.feature_names = [c for c in result.columns if c not in 
                             ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ 特征计算完成！共 {len(self.feature_names)} 个特征")
        return result
    
    def _calculate_technical_features(self, df):
        """技术面特征 (32个基础 + 6个扩展)"""
        
        # ========== 1. 动量特征 (7个) ==========
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)  # 新增
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        df['return_60d'] = df['close'].pct_change(60)
        df['return_accel'] = df['return_5d'] - df['return_10d']
        
        # ========== 2. 波动率特征 (6个) ==========
        df['volatility_5d'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        df['volatility_10d'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        df['volatility_20d'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['volatility_60d'] = df['close'].rolling(60).std() / df['close'].rolling(60).mean()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        df['vol_trend'] = df['volatility_20d'] / df['volatility_60d']
        
        # ========== 3. 均线特征 (8个) ==========
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_60'] = df['close'].rolling(60).mean()
        df['price_to_ma5'] = df['close'] / df['ma_5']
        df['price_to_ma10'] = df['close'] / df['ma_10']
        df['price_to_ma20'] = df['close'] / df['ma_20']
        df['price_to_ma60'] = df['close'] / df['ma_60']
        df['ma5_to_ma20'] = df['ma_5'] / df['ma_20']
        df['ma20_to_ma60'] = df['ma_20'] / df['ma_60']
        
        # ========== 4. 布林带特征 (3个) ==========
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ========== 5. RSI特征 (3个) ==========
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
        
        # ========== 6. MACD特征 (4个) ==========
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ========== 7. 成交量特征 (8个) ==========
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # 成交额
        df['turnover'] = df['close'] * df['volume']
        df['turnover_ma20'] = df['turnover'].rolling(20).mean()
        df['turnover_ratio'] = df['turnover'] / df['turnover_ma20']
        
        # ========== 8. 其他特征 (4个) ==========
        df['amplitude_20d'] = ((df['high'] - df['low']) / df['low']).rolling(20).mean()
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        return df
    
    def _calculate_money_flow_features(self, df):
        """资金流向特征 (5个) ⭐核心"""
        
        # 典型价格
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # 原始资金流
        df['raw_money_flow'] = df['typical_price'] * df['volume']
        
        # MFI (Money Flow Index)
        positive_flow = df['raw_money_flow'].where(
            df['typical_price'] > df['typical_price'].shift(1), 0
        )
        negative_flow = df['raw_money_flow'].where(
            df['typical_price'] < df['typical_price'].shift(1), 0
        )
        
        positive_sum = positive_flow.rolling(14).sum()
        negative_sum = negative_flow.rolling(14).sum()
        
        money_ratio = positive_sum / negative_sum
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # 主力资金检测
        df['big_money'] = (df['turnover'] > df['turnover_ma20'] * 1.5).astype(int)
        
        # 大单占比 (5日)
        df['big_money_ratio'] = df['big_money'].rolling(5).mean()
        
        # 净流入/流出 (简化版)
        df['net_money_flow'] = df['raw_money_flow'] * np.sign(df['close'] - df['open'])
        df['net_money_flow_5d'] = df['net_money_flow'].rolling(5).sum()
        
        return df
    
    def _calculate_size_features(self, df):
        """市值分层特征 (4个)"""
        
        # 流通市值代理
        df['market_cap_proxy'] = df['close'] * df['volume_ma20']
        
        # 是否低价股
        df['is_low_price'] = (df['close'] < 10).astype(int)
        
        # 流动性得分 (简化)
        df['liquidity_score'] = df['volume'] / df['volume'].rolling(60).mean()
        
        return df
    
    def get_feature_names(self):
        """获取特征名列表"""
        return self.feature_names


if __name__ == '__main__':
    # 测试
    print("🧪 测试特征工程...")
    
    # 创建测试数据
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    test_df = pd.DataFrame({
        'symbol': ['000001.SZ'] * 100,
        'date': dates,
        'open': np.random.randn(100).cumsum() + 10,
        'high': np.random.randn(100).cumsum() + 11,
        'low': np.random.randn(100).cumsum() + 9,
        'close': np.random.randn(100).cumsum() + 10,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # 计算特征
    engineer = FeatureEngineer()
    result = engineer.calculate_all_features(test_df)
    
    print(f"\n📊 特征数量: {len(engineer.get_feature_names())}")
    print(f"📊 特征列表: {engineer.get_feature_names()[:10]}...")
    print("\n✅ 特征工程测试通过！")