#!/usr/bin/env python3
"""
回测模型 V3 - 简化正确版
- 考虑交易成本
- 持仓5天
- T+1交易逻辑
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(model_data, X):
    models = model_data['models']
    weights = model_data['weights']
    predictions = []
    for name, model in models.items():
        if name in weights and weights[name] > 0:
            pred = model.predict(X)
            predictions.append(weights[name] * pred)
    return np.sum(predictions, axis=0)

def backtest_v3():
    print("="*60)
    print("🚀 V3回测 - 简化正确版")
    print("="*60)
    
    # 加载数据
    data_file = Path("data/processed/valid_features.csv")
    model_path = Path("model/model_csi500.pkl")
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2025-01-01']
    
    print(f"数据: {len(df)} 条, {df['date'].nunique()} 天")
    
    # 加载模型预测
    model_data = load_model(model_path)
    features = [f for f in model_data['features'] if f in df.columns]
    df['pred'] = predict(model_data, df[features].values)
    
    # T+1回测：今天预测，明天买入，持有5天
    dates = sorted(df['date'].unique())
    trades = []
    
    for i in range(len(dates) - 6):  # 留出5天持仓+1天下一天
        today = dates[i]
        tomorrow = dates[i+1]
        sell_day = dates[i+6]  # 持有5天后卖出
        
        # 今天选股
        today_df = df[df['date'] == today]
        tomorrow_df = df[df['date'] == tomorrow][['symbol', 'open']].rename(columns={'open': 'buy_open'})
        sell_df = df[df['date'] == sell_day][['symbol', 'close']].rename(columns={'close': 'sell_close'})
        
        # 选前5%
        n_select = max(1, int(len(today_df) * 0.05))
        top_stocks = today_df.nlargest(n_select, 'pred')
        
        # 合并买入和卖出数据
        merged = top_stocks.merge(tomorrow_df, on='symbol').merge(sell_df, on='symbol')
        
        if len(merged) == 0:
            continue
        
        # 计算收益（考虑成本）
        # 买入成本：价格 * 1.0013（滑点0.1% + 手续费0.03%）
        # 卖出成本：价格 * 0.9987
        buy_price = merged['buy_open'] * 1.0013
        sell_price = merged['sell_close'] * 0.9987
        returns = (sell_price / buy_price) - 1
        
        for j, row in merged.iterrows():
            trades.append({
                'buy_date': tomorrow.strftime('%Y-%m-%d'),
                'sell_date': sell_day.strftime('%Y-%m-%d'),
                'symbol': row['symbol'],
                'return': returns[j],
                'holding_days': 5
            })
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ 无交易")
        return
    
    # 统计
    avg_return = trades_df['return'].mean()
    win_rate = (trades_df['return'] > 0).mean()
    
    # 假设每周交易一次（5天持仓），年化
    weekly_return = avg_return
    annual_return = (1 + weekly_return) ** 52 - 1
    
    # 成本：单次0.13%，买卖双向0.26%，每周一次年化13.5%
    annual_cost = 0.0026 * 52
    
    print(f"\n{'='*60}")
    print("📊 V3回测结果")
    print(f"{'='*60}")
    print(f"总交易: {len(trades_df)} 笔")
    print(f"平均收益: {avg_return*100:.3f}%")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"年化收益(毛): {annual_return*100:.2f}%")
    print(f"年化成本: {annual_cost*100:.2f}%")
    print(f"年化收益(净): {(annual_return-annual_cost)*100:.2f}%")
    print(f"平均持仓: {trades_df['holding_days'].mean():.1f} 天")
    
    # 保存
    trades_df.to_csv('backtest_result/trades_v3.csv', index=False)
    print(f"\n✅ 保存: backtest_result/trades_v3.csv")

if __name__ == '__main__':
    backtest_v3()
