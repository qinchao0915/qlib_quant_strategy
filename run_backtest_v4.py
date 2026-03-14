#!/usr/bin/env python3
"""
回测模型 V4 - 最终版
- MA20大盘过滤器
- -8%强制止损
- 5天持仓
- 使用V2模型
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


def calculate_market_filter(df, current_date):
    """
    大盘过滤器：基于指数MA20
    返回: True(可以交易) / False(空仓)
    """
    # 获取指数数据（用成分股平均作为代理）
    idx_data = df[df['date'] <= current_date].groupby('date')['close'].mean().reset_index()
    idx_data = idx_data.tail(30)  # 最近30天
    
    if len(idx_data) < 20:
        return True  # 数据不足，默认允许交易
    
    # 计算MA20
    ma20 = idx_data['close'].rolling(20).mean().iloc[-1]
    current = idx_data['close'].iloc[-1]
    
    # 价格在MA20上方才交易
    return current > ma20


def backtest_v4():
    print("="*70)
    print("🚀 V4回测 - 最终版")
    print("特点: MA20过滤 + -8%止损 + 5天持仓 + V2模型")
    print("="*70)
    
    # 加载V2模型
    model_path = Path("model/model_csi500_v2.pkl")
    if not model_path.exists():
        print(f"❌ V2模型不存在: {model_path}")
        print("请先运行: python3 train_model_v2.py")
        return
    
    model_data = load_model(model_path)
    features = model_data['features']
    print(f"✅ 加载V2模型 (IC: {model_data.get('ensemble_ic', 'N/A'):.4f})")
    
    # 加载数据
    data_file = Path("data/processed/valid_features.csv")
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2025-01-01']
    
    print(f"✅ 数据: {len(df)} 条, {df['date'].nunique()} 天")
    
    # 预测
    features = [f for f in features if f in df.columns]
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 参数设置
    HOLDING_DAYS = 5
    TOP_PCT = 0.05  # 选股前5%
    STOP_LOSS = -0.08  # -8%止损
    
    # T+1回测
    dates = sorted(df['date'].unique())
    trades = []
    daily_stats = []
    
    # 持仓管理: symbol -> {buy_date, buy_price, max_price}
    positions = {}
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1]
        
        today_df = df[df['date'] == today]
        tomorrow_df = df[df['date'] == tomorrow]
        
        # ===== 1. 检查止损 =====
        for symbol, pos in list(positions.items()):
            stock_today = today_df[today_df['symbol'] == symbol]
            if len(stock_today) == 0:
                continue
            
            current_price = stock_today['close'].values[0]
            buy_price = pos['buy_price']
            current_return = (current_price / buy_price) - 1
            
            # 更新最高价（用于移动止损）
            if current_price > pos.get('max_price', buy_price):
                pos['max_price'] = current_price
            
            # -8%强制止损
            if current_return <= STOP_LOSS:
                # 明天开盘卖出
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * 0.9987  # 扣除成本
                    ret = (sell_price / buy_price) - 1
                    
                    trades.append({
                        'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                        'sell_date': tomorrow.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': (tomorrow - pos['buy_date']).days,
                        'reason': 'stop_loss'
                    })
                    del positions[symbol]
        
        # ===== 2. 检查持仓到期 =====
        for symbol, pos in list(positions.items()):
            if (today - pos['buy_date']).days >= HOLDING_DAYS:
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * 0.9987
                    ret = (sell_price / pos['buy_price']) - 1
                    
                    trades.append({
                        'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                        'sell_date': tomorrow.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': pos['buy_price'],
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': HOLDING_DAYS,
                        'reason': 'holding_end'
                    })
                    del positions[symbol]
        
        # ===== 3. 大盘过滤 =====
        can_trade = calculate_market_filter(df, today)
        
        if not can_trade:
            # 空仓日，卖出所有持仓
            for symbol, pos in list(positions.items()):
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * 0.9987
                    ret = (sell_price / pos['buy_price']) - 1
                    
                    trades.append({
                        'buy_date': pos['buy_date'].strftime('%Y-%m-%d'),
                        'sell_date': tomorrow.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': pos['buy_price'],
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': (tomorrow - pos['buy_date']).days,
                        'reason': 'market_filter'
                    })
                    del positions[symbol]
            
            daily_stats.append({
                'date': today,
                'n_positions': 0,
                'can_trade': False
            })
            continue
        
        # ===== 4. 买入新股票 =====
        # 选前5%
        n_select = max(1, int(len(today_df) * TOP_PCT))
        
        # 排除已有持仓
        available = today_df[~today_df['symbol'].isin(positions.keys())]
        
        if len(available) >= n_select:
            top_stocks = available.nlargest(n_select, 'pred')
            
            for _, stock in top_stocks.iterrows():
                symbol = stock['symbol']
                
                # 明天开盘买入
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    buy_price = stock_tomorrow['open'].values[0] * 1.0013  # 加上成本
                    
                    positions[symbol] = {
                        'buy_date': tomorrow,
                        'buy_price': buy_price,
                        'max_price': buy_price,
                        'pred': stock['pred']
                    }
        
        daily_stats.append({
            'date': today,
            'n_positions': len(positions),
            'can_trade': True
        })
    
    # 处理最后未卖出的持仓
    if len(dates) > 0 and len(positions) > 0:
        last_date = dates[-1]
        last_df = df[df['date'] == last_date]
        
        for symbol, pos in positions.items():
            stock_last = last_df[last_df['symbol'] == symbol]
            if len(stock_last) > 0:
                sell_price = stock_last['close'].values[0] * 0.9987
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
        print("❌ 无交易数据")
        return
    
    # 统计
    avg_return = trades_df['return'].mean()
    win_rate = (trades_df['return'] > 0).mean()
    
    # 年化（假设每周交易一次）
    weekly_return = avg_return
    annual_return = (1 + weekly_return) ** 52 - 1
    annual_cost = 0.0026 * 52  # 0.26%每周
    
    # 止损统计
    stop_loss_count = (trades_df['reason'] == 'stop_loss').sum()
    
    print(f"\n{'='*70}")
    print("📊 V4回测结果")
    print(f"{'='*70}")
    print(f"总交易: {len(trades_df)} 笔")
    print(f"平均收益: {avg_return*100:.3f}%")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"年化收益(毛): {annual_return*100:.2f}%")
    print(f"年化成本: {annual_cost*100:.2f}%")
    print(f"年化收益(净): {(annual_return-annual_cost)*100:.2f}%")
    print(f"平均持仓: {trades_df['holding_days'].mean():.1f} 天")
    print(f"\n止损统计:")
    print(f"  -8%止损: {stop_loss_count} 笔 ({stop_loss_count/len(trades_df)*100:.1f}%)")
    print(f"  持仓到期: {(trades_df['reason']=='holding_end').sum()} 笔")
    print(f"  大盘过滤: {(trades_df['reason']=='market_filter').sum()} 笔")
    
    # 保存
    output_dir = Path("backtest_result")
    output_dir.mkdir(exist_ok=True)
    trades_df.to_csv(output_dir / "trades_v4_final.csv", index=False)
    print(f"\n✅ 保存: backtest_result/trades_v4_final.csv")


if __name__ == '__main__':
    backtest_v4()
