#!/usr/bin/env python3
"""
回测模型 V2 - 改进版（修复版）
- 考虑交易成本
- 添加大盘过滤器
- 持仓时间至少5天
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


def calculate_transaction_cost(price, is_buy=True):
    """计算交易成本"""
    commission_rate = 0.0003  # 万分之3
    slippage = 0.001  # 0.1%
    total_cost_rate = commission_rate + slippage
    
    if is_buy:
        return price * (1 + total_cost_rate)
    else:
        return price * (1 - total_cost_rate)


def backtest_index_v2(index_name, model_path, data_dir, 
                      start_date='2025-01-01', end_date='2025-12-31',
                      holding_days=5, top_pct=0.05):
    """V2改进版回测"""
    print(f"\n{'='*80}")
    print(f"📊 回测 {index_name.upper()} - V2改进版")
    print(f"{'='*80}")
    print(f"参数: 持仓{holding_days}天, 选股前{top_pct*100}%, 考虑交易成本")
    
    # 加载模型
    model_data = load_model(model_path)
    features = model_data['features']
    
    # 加载数据
    if index_name == 'csi500':
        data_file = data_dir / "valid_features.csv"
    else:
        data_file = data_dir / index_name / "valid_features.csv"
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    print(f"✅ 加载数据: {len(df)} 条记录, {df['date'].nunique()} 个交易日")
    
    # 预测
    features = [f for f in features if f in df.columns]
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 初始化
    positions = {}  # symbol -> {buy_price, buy_date, pred}
    trades = []
    daily_portfolio = []
    
    dates = sorted(df['date'].unique())
    
    for i, current_date in enumerate(dates):
        current_df = df[df['date'] == current_date].copy()
        
        # 获取未来holding_days天的数据
        future_idx = i + holding_days
        if future_idx >= len(dates):
            break
        
        sell_date = dates[future_idx]
        sell_df = df[df['date'] == sell_date][['symbol', 'open', 'close']].copy()
        sell_df = sell_df.rename(columns={'open': 'sell_open', 'close': 'sell_close'})
        
        # 简单大盘过滤：用当日涨跌判断
        market_return = current_df['pct_chg'].mean() if 'pct_chg' in current_df.columns else 0
        
        # 空仓条件：大盘大跌（-2%以上）
        if market_return < -2:
            # 卖出所有持仓
            for symbol, pos in list(positions.items()):
                sell_data = sell_df[sell_df['symbol'] == symbol]
                if len(sell_data) > 0:
                    sell_price = calculate_transaction_cost(
                        sell_data['sell_open'].values[0], is_buy=False
                    )
                    ret = (sell_price / pos['buy_price']) - 1
                    trades.append({
                        'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                        'sell_date': sell_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': pos['buy_price'],
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': holding_days,
                        'reason': 'market_exit'
                    })
                    del positions[symbol]
            continue
        
        # 检查是否有持仓到期需要卖出
        for symbol, pos in list(positions.items()):
            if (current_date - pos['buy_date']).days >= holding_days:
                sell_data = sell_df[sell_df['symbol'] == symbol]
                if len(sell_data) > 0:
                    sell_price = calculate_transaction_cost(
                        sell_data['sell_open'].values[0], is_buy=False
                    )
                    ret = (sell_price / pos['buy_price']) - 1
                    trades.append({
                        'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                        'sell_date': sell_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': pos['buy_price'],
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': holding_days,
                        'reason': 'holding_end'
                    })
                    del positions[symbol]
        
        # 买入新股票
        n_select = max(1, int(len(current_df) * top_pct))
        
        if len(positions) < n_select:
            # 选预测得分最高的
            available = current_df[~current_df['symbol'].isin(positions.keys())]
            if len(available) > 0:
                top_stocks = available.nlargest(n_select - len(positions), 'pred')
                
                for _, stock in top_stocks.iterrows():
                    symbol = stock['symbol']
                    if symbol not in positions:
                        buy_price = calculate_transaction_cost(stock['open'], is_buy=True)
                        positions[symbol] = {
                            'buy_price': buy_price,
                            'buy_date': current_date,
                            'pred': stock['pred']
                        }
        
        # 记录当日持仓
        daily_portfolio.append({
            'date': current_date,
            'n_positions': len(positions),
            'symbols': list(positions.keys())
        })
    
    # 处理最后未卖出的持仓
    if len(dates) > 0:
        last_date = dates[-1]
        last_df = df[df['date'] == last_date]
        for symbol, pos in positions.items():
            sell_data = last_df[last_df['symbol'] == symbol]
            if len(sell_data) > 0:
                sell_price = calculate_transaction_cost(sell_data['close'].values[0], is_buy=False)
                ret = (sell_price / pos['buy_price']) - 1
                trades.append({
                    'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                    'sell_date': last_date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'buy_price': pos['buy_price'],
                    'sell_price': sell_price,
                    'return': ret,
                    'holding_days': (last_date - pos['buy_date']).days,
                    'reason': 'end_of_period'
                })
    
    # 分析结果
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ 没有交易数据")
        return None
    
    # 计算收益（简单平均，不复利）
    avg_return = trades_df['return'].mean()
    total_trades = len(trades_df)
    
    # 年化收益（假设每周交易一次）
    weekly_return = avg_return
    annual_return = (1 + weekly_return) ** 52 - 1
    
    # 胜率
    win_rate = (trades_df['return'] > 0).mean()
    
    # 成本
    total_cost = total_trades * 0.0013 * 2  # 单次0.13%，买卖双向
    
    print(f"\n{'='*80}")
    print(f"📊 {index_name.upper()} V2回测结果")
    print(f"{'='*80}")
    print(f"总交易次数:   {total_trades} 笔")
    print(f"平均单笔收益: {avg_return*100:.3f}%")
    print(f"预估年化收益: {annual_return*100:.2f}%")
    print(f"胜率:         {win_rate*100:.1f}%")
    print(f"总交易成本:   {total_cost*100:.2f}%")
    print(f"净年化收益:   {(annual_return-total_cost)*100:.2f}%")
    print(f"平均持仓天数: {trades_df['holding_days'].mean():.1f} 天")
    print(f"盈利交易:     {(trades_df['return'] > 0).sum()} 笔")
    print(f"亏损交易:     {(trades_df['return'] < 0).sum()} 笔")
    
    # 保存结果
    output_dir = Path("backtest_result")
    output_dir.mkdir(exist_ok=True)
    trades_df.to_csv(output_dir / f"trades_{index_name}_2025_v2.csv", index=False)
    
    print(f"\n✅ 结果保存: {output_dir}/trades_{index_name}_2025_v2.csv")
    
    return trades_df


def main():
    print("="*80)
    print("🚀 V2改进版回测（修复版）")
    print("="*80)
    
    data_dir = Path("data/processed")
    model_dir = Path("model")
    
    result = backtest_index_v2(
        'csi500', 
        model_dir / "model_csi500.pkl", 
        data_dir,
        holding_days=5,
        top_pct=0.05
    )
    
    if result is not None:
        print("\n✅ 回测完成！")


if __name__ == '__main__':
    main()