#!/usr/bin/env python3
"""
交易工具

功能：
- 仓位管理
- 风险控制
- 交易信号生成
"""

import pandas as pd
import numpy as np
from datetime import datetime


class TradingUtils:
    """交易工具类"""
    
    @staticmethod
    def generate_signals(pred_df, threshold=0.02):
        """
        生成交易信号
        
        Args:
            pred_df: 预测结果 DataFrame，需包含 'score' 列
            threshold: 买入阈值
            
        Returns:
            DataFrame: 添加 'signal' 列
        """
        df = pred_df.copy()
        
        # 根据预测分数生成信号
        df['signal'] = 0
        df.loc[df['score'] > threshold, 'signal'] = 1   # 买入
        df.loc[df['score'] < -threshold, 'signal'] = -1 # 卖出
        
        return df
    
    @staticmethod
    def select_top_stocks(pred_df, top_n=10, date_col='date', score_col='score'):
        """
        选择Top N股票
        
        Args:
            pred_df: 预测结果
            top_n: 选择数量
            date_col: 日期列名
            score_col: 分数列名
            
        Returns:
            DataFrame: 每日Top N股票
        """
        results = []
        
        for date, group in pred_df.groupby(date_col):
            # 按分数排序，取Top N
            top_stocks = group.nlargest(top_n, score_col)
            top_stocks['rank'] = range(1, len(top_stocks) + 1)
            results.append(top_stocks)
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def calculate_position_size(capital, price, risk_pct=0.02):
        """
        计算仓位大小
        
        Args:
            capital: 总资金
            price: 股票价格
            risk_pct: 风险比例
            
        Returns:
            int: 买入股数（100的整数倍，A股规定）
        """
        max_amount = capital * risk_pct
        shares = int(max_amount / price / 100) * 100  # A股100股为单位
        return max(shares, 0)
    
    @staticmethod
    def save_recommendation(selected_df, output_file):
        """
        保存选股推荐
        
        Args:
            selected_df: 选股结果
            output_file: 输出文件路径
        """
        # 添加时间戳
        selected_df['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存为CSV
        selected_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"选股推荐已保存: {output_file}")
        
        # 打印摘要
        print("\n📊 选股摘要:")
        print(f"  日期: {selected_df['date'].iloc[0] if 'date' in selected_df.columns else 'N/A'}")
        print(f"  选股数量: {len(selected_df)}")
        if 'symbol' in selected_df.columns:
            print(f"  股票列表: {', '.join(selected_df['symbol'].tolist())}")


if __name__ == '__main__':
    # 测试
    data = {
        'date': ['2024-03-01'] * 5,
        'symbol': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
        'score': [0.05, 0.03, 0.02, -0.01, -0.03]
    }
    df = pd.DataFrame(data)
    
    # 生成信号
    df = TradingUtils.generate_signals(df, threshold=0.01)
    print("交易信号:")
    print(df)
    
    # 选择Top 3
    top3 = TradingUtils.select_top_stocks(df, top_n=3)
    print("\nTop 3 股票:")
    print(top3)
