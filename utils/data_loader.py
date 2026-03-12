#!/usr/bin/env python3
"""
数据加载工具

功能：
- 从 Tushare 获取 A股数据
- 转换为 Qlib 格式
- 数据清洗和预处理
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataLoader:
    """数据加载器"""
    
    def __init__(self, tushare_token=None):
        """
        初始化
        
        Args:
            tushare_token: Tushare API Token，默认从环境变量读取
        """
        self.token = tushare_token or os.getenv('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("请设置 TUSHARE_TOKEN 环境变量或在初始化时传入")
        
        # 延迟导入，避免未安装时报错
        try:
            import tushare as ts
            self.ts = ts
            self.ts.set_token(self.token)
            self.pro = self.ts.pro_api()
        except ImportError:
            raise ImportError("请先安装 tushare: pip install tushare")
    
    def get_daily_data(self, ts_code, start_date, end_date):
        """
        获取日线数据
        
        Args:
            ts_code: 股票代码，如 '000001.SZ'
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            
        Returns:
            DataFrame: 日线数据
        """
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return pd.DataFrame()
        
        # 数据清洗
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_stock_list(self, exchange=''):
        """
        获取股票列表
        
        Args:
            exchange: 交易所 SSE/SZSE，空字符串表示全部
            
        Returns:
            DataFrame: 股票列表
        """
        df = self.pro.stock_basic(exchange=exchange, list_status='L')
        return df
    
    def to_qlib_format(self, df, symbol_field='ts_code', date_field='trade_date'):
        """
        转换为 Qlib 格式
        
        Qlib 需要的列：
        - date: 日期
        - symbol: 股票代码
        - $open, $high, $low, $close, $volume: 价格和成交量
        """
        if df.empty:
            return pd.DataFrame()
        
        # 重命名列
        qlib_df = df.copy()
        
        # 确保必要的列存在
        column_mapping = {
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'close': '$close',
            'vol': '$volume',
            'amount': '$amount'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in qlib_df.columns:
                qlib_df.rename(columns={old_col: new_col}, inplace=True)
        
        return qlib_df


if __name__ == '__main__':
    # 测试
    loader = DataLoader()
    
    # 获取单只股票数据
    df = loader.get_daily_data('000001.SZ', '20240101', '20240312')
    print(f"获取到 {len(df)} 条数据")
    print(df.head())
