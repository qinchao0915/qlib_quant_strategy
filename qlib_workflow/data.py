#!/usr/bin/env python3
"""
Qlib Data 组件

功能：
- 数据加载
- 数据转换
- 数据缓存
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


class QlibData:
    """Qlib 风格数据管理器"""
    
    def __init__(self, provider_uri, cache_path="data/qlib_cache"):
        """
        初始化
        
        Args:
            provider_uri: 数据提供者（如 Tushare）
            cache_path: 缓存路径
        """
        self.provider = provider_uri
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
    
    def load_stock_data(self, symbols, start_date, end_date, fields=None):
        """
        加载股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            
        Returns:
            DataFrame: Qlib 格式数据
        """
        cache_file = self.cache_path / f"qlib_data_{start_date}_{end_date}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            print(f"📂 从缓存加载数据")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 从提供者获取数据
        print(f"📥 从 {self.provider} 获取数据...")
        # 这里调用 TushareProvider
        
        # 转换为 Qlib 格式
        df = self._convert_to_qlib_format(df)
        
        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        return df
    
    def _convert_to_qlib_format(self, df):
        """
        转换为 Qlib 格式
        
        Qlib 格式要求：
        - index: datetime
        - columns: 特征字段
        - 支持多股票（通过 symbol 列）
        """
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        
        # 设置索引
        df = df.set_index(['date', 'symbol'])
        
        return df
    
    def dump(self, df, name):
        """保存数据"""
        file_path = self.cache_path / f"{name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"💾 数据已保存: {file_path}")
    
    def load(self, name):
        """加载数据"""
        file_path = self.cache_path / f"{name}.pkl"
        with open(file_path, 'rb') as f:
            return pickle.load(f)
