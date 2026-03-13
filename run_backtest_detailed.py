#!/usr/bin/env python3
"""
回测模型 - 输出个股交易明细
"""

import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_model(model_path):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict(model_data, X):
    """使用模型进行预测"""
    models = model_data['models']
    weights = model_data['weights']
    
    predictions = []
    for name, model in models.items():
        if name in weights and weights[name] > 0:
            pred = model.predict(X)
            predictions.append(weights[name] * pred)
    
    return np.sum(predictions, axis=0)


def backtest_index_detailed(index_name, model_path, data_dir, start_date='2025-01-01', end_date='2025-12-31'):
    """回测单个指数 - 输出个股明细"""
    print(f"\n{'='*80}")
    print(f"📊 回测 {index_name.upper()} - 个股交易明细")
    print(f"{'='*80}")
    
    # 加载模型
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model_data = load_model(model_path)
    features = model_data['features']
    
    # 加载数据
    if index_name == 'csi500':
        data_file = data_dir / "valid_features.csv"
    else:
        data_file = data_dir / index_name / "valid_features.csv"
    
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if df.empty:
        print(f"❌ 没有找到数据")
        return None
    
    print(f"✅ 加载数据: {len(df)} 条记录, {df['date'].nunique()} 个交易日")
    
    # 确保所有特征都存在
    features = [f for f in features if f in df.columns]
    
    # 预测
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # T+1回测 - 生成个股交易明细
    print(f"\n📈 生成个股交易明细...")
    
    trades = []  # 存储所有交易记录
    dates = sorted(df['date'].unique())
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # 获取当前日期的数据
        current_df = df[df['date'] == current_date].copy()
        
        # 获取次日数据
        next_df = df[df['date'] == next_date][['symbol', 'open', 'close', 'pre_close']].copy()
        next_df = next_df.rename(columns={
            'open': 'next_open', 
            'close': 'next_close',
            'pre_close': 'next_pre_close'
        })
        
        # 合并数据
        merged = current_df.merge(next_df, on='symbol', how='inner')
        
        if len(merged) == 0:
            continue
        
        # 按预测值排序，选择前20%的股票
        n_select = max(1, int(len(merged) * 0.2))
        top_stocks = merged.nlargest(n_select, 'pred')
        
        # 为每只股票生成交易记录
        for _, stock in top_stocks.iterrows():
            trade = {
                'trade_date': next_date.strftime('%Y-%m-%d'),  # 实际交易日期（T+1）
                'signal_date': current_date.strftime('%Y-%m-%d'),  # 信号生成日期（T）
                'symbol': stock['symbol'],
                'pred_score': stock['pred'],
                'buy_price': stock['next_open'],  # T+1开盘价买入
                'sell_price': stock['next_close'],  # T+1收盘价卖出
                'prev_close': stock['pre_close'],  # T日收盘价（用于参考）
                'return': (stock['next_close'] / stock['next_open']) - 1,  # 当日收益率
                'holding_period': '1天',  # 日内交易
                'buy_signal': f'预测得分{stock["pred"]:.6f}排名前20%',
                'sell_signal': '收盘卖出',
                'position_pct': 1.0 / n_select  # 等权重持仓
            }
            trades.append(trade)
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ 没有生成交易记录")
        return None
    
    # 计算每日组合收益
    daily_returns = trades_df.groupby('trade_date').agg({
        'return': 'mean',
        'symbol': 'count'
    }).reset_index()
    daily_returns.columns = ['date', 'portfolio_return', 'n_stocks']
    daily_returns['cum_return'] = (1 + daily_returns['portfolio_return']).cumprod() - 1
    
    # 保存结果
    output_dir = Path("backtest_result")
    output_dir.mkdir(exist_ok=True)
    
    trades_file = output_dir / f"trades_{index_name}_2025_detailed.csv"
    daily_file = output_dir / f"daily_{index_name}_2025.csv"
    
    trades_df.to_csv(trades_file, index=False)
    daily_returns.to_csv(daily_file, index=False)
    
    print(f"\n✅ 交易明细保存: {trades_file}")
    print(f"✅ 每日收益保存: {daily_file}")
    
    # 打印汇总
    print(f"\n{'='*80}")
    print(f"📊 {index_name.upper()} 交易汇总")
    print(f"{'='*80}")
    print(f"总交易次数: {len(trades_df)} 笔")
    print(f"交易天数: {trades_df['trade_date'].nunique()} 天")
    print(f"平均每交易日持仓: {len(trades_df) / trades_df['trade_date'].nunique():.0f} 只")
    print(f"盈利交易: {(trades_df['return'] > 0).sum()} 笔 ({(trades_df['return'] > 0).mean()*100:.1f}%)")
    print(f"亏损交易: {(trades_df['return'] < 0).sum()} 笔 ({(trades_df['return'] < 0).mean()*100:.1f}%)")
    print(f"平均单笔收益: {trades_df['return'].mean()*100:.3f}%")
    print(f"最大单笔盈利: {trades_df['return'].max()*100:.2f}%")
    print(f"最大单笔亏损: {trades_df['return'].min()*100:.2f}%")
    
    total_return = daily_returns['cum_return'].iloc[-1]
    print(f"\n组合总收益: {total_return*100:.2f}%")
    
    return trades_df, daily_returns


def main():
    """主函数"""
    print("="*80)
    print("🚀 生成个股交易明细")
    print("="*80)
    
    data_dir = Path("data/processed")
    model_dir = Path("model")
    
    # 生成CSI500的交易明细
    index_name = 'csi500'
    model_path = model_dir / f"model_{index_name}.pkl"
    
    result = backtest_index_detailed(index_name, model_path, data_dir)
    
    if result:
        trades_df, daily_returns = result
        
        # 展示前20笔交易
        print(f"\n{'='*80}")
        print("📋 前20笔交易明细")
        print(f"{'='*80}")
        
        display_cols = ['trade_date', 'symbol', 'pred_score', 'buy_price', 'sell_price', 'return', 'buy_signal']
        print(trades_df[display_cols].head(20).to_string(index=False))
        
        # 展示收益最高的5笔交易
        print(f"\n{'='*80}")
        print("🏆 收益最高的5笔交易")
        print(f"{'='*80}")
        top_trades = trades_df.nlargest(5, 'return')[display_cols]
        print(top_trades.to_string(index=False))
        
        # 展示亏损最大的5笔交易
        print(f"\n{'='*80}")
        print("📉 亏损最大的5笔交易")
        print(f"{'='*80}")
        bottom_trades = trades_df.nsmallest(5, 'return')[display_cols]
        print(bottom_trades.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✅ 完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
