#!/usr/bin/env python3
"""
Qlib Strategy 组件

功能：
- 策略生成
- 信号生成
- 权重优化
"""

import pandas as pd
import numpy as np


class QlibStrategy:
    """Qlib 风格策略"""
    
    def __init__(self, model, topk=50, drop=0):
        """
        初始化
        
        Args:
            model: 训练好的模型
            topk: 选股数量
            drop: 剔除数量
        """
        self.model = model
        self.topk = topk
        self.drop = drop
    
    def generate_signals(self, df, features):
        """
        生成交易信号
        
        Args:
            df: 特征数据
            features: 特征列名
            
        Returns:
            DataFrame: 包含预测分数和信号
        """
        # 预测
        X = df[features]
        df['score'] = self.model.predict(X)
        
        # 生成信号
        signals = []
        for date, group in df.groupby(level='date'):
            # 排序
            group = group.sort_values('score', ascending=False)
            
            # 选 Top K
            group['signal'] = 0
            group.iloc[self.drop:self.drop+self.topk, group.columns.get_loc('signal')] = 1
            
            signals.append(group)
        
        return pd.concat(signals)
    
    def get_daily_selection(self, df, date=None):
        """
        获取每日选股
        
        Args:
            df: 信号数据
            date: 日期（默认最新）
            
        Returns:
            DataFrame: 选股结果
        """
        if date is None:
            date = df.index.get_level_values('date').max()
        
        day_df = df.loc[df.index.get_level_values('date') == date]
        selected = day_df[day_df['signal'] == 1].sort_values('score', ascending=False)
        
        return selected[['score', 'signal']]
