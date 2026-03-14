#!/usr/bin/env python3
"""
V2模型最终回测
- 使用V2模型
- 数据穿越检查
- 包含股票名称
- 保存在result文件夹
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_stock_names():
    """加载股票名称映射"""
    # 优先使用完整映射
    name_file = Path('data/cache/stock_name_map_full.json')
    if not name_file.exists():
        name_file = Path('data/cache/stock_name_map.json')
    
    if name_file.exists():
        with open(name_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def predict(model_data, X):
    models = model_data['models']
    weights = model_data['weights']
    predictions = []
    for name, model in models.items():
        if name in weights and weights[name] > 0:
            pred = model.predict(X)
            predictions.append(weights[name] * pred)
    return np.sum(predictions, axis=0)


def check_data_leakage(train_df, valid_df):
    """检查数据穿越"""
    print("\n" + "="*70)
    print("🔍 数据穿越检查")
    print("="*70)
    
    train_dates = set(train_df['date'].unique())
    valid_dates = set(valid_df['date'].unique())
    overlap = train_dates & valid_dates
    
    if overlap:
        print(f"❌ 发现数据穿越！重叠日期: {len(overlap)} 天")
        print(f"   示例: {sorted(list(overlap))[:5]}")
        return False
    else:
        print(f"✅ 无数据穿越")
        print(f"   训练集: {train_df['date'].min()} ~ {train_df['date'].max()}")
        print(f"   验证集: {valid_df['date'].min()} ~ {valid_df['date'].max()}")
        return True


def backtest_v2_final():
    print("="*70)
    print("🚀 V2模型最终回测")
    print("="*70)
    
    # 加载V2模型
    model_path = Path("model/model_csi500_v2.pkl")
    if not model_path.exists():
        print(f"❌ V2模型不存在: {model_path}")
        return
    
    model_data = load_model(model_path)
    features = model_data['features']
    print(f"✅ 加载V2模型")
    print(f"   模型IC: {model_data.get('ensemble_ic', 'N/A'):.4f}")
    print(f"   特征数: {len(features)}")
    print(f"   训练样本: {model_data.get('train_samples', 'N/A')}")
    print(f"   验证样本: {model_data.get('valid_samples', 'N/A')}")
    
    # 加载股票名称
    stock_names = load_stock_names()
    print(f"✅ 加载股票名称映射: {len(stock_names)} 只")
    
    # 加载数据
    train_file = Path("data/processed/train_features.csv")
    valid_file = Path("data/processed/valid_features.csv")
    
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    train_df['date'] = pd.to_datetime(train_df['date'])
    valid_df['date'] = pd.to_datetime(valid_df['date'])
    
    # 数据穿越检查
    if not check_data_leakage(train_df, valid_df):
        print("❌ 数据穿越检查失败，停止回测")
        return
    
    # 只使用验证集（2024-09起）
    df = valid_df.copy()
    df = df[df['date'] >= '2025-01-01']  # 2025年回测
    
    print(f"\n✅ 回测数据: {len(df)} 条, {df['date'].nunique()} 天")
    print(f"   股票数: {df['symbol'].nunique()}")
    print(f"   日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 预测
    features = [f for f in features if f in df.columns]
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 添加股票名称
    df['stock_name'] = df['symbol'].map(stock_names)
    df['stock_name'] = df['stock_name'].fillna('未知')
    
    # T+1回测参数
    HOLDING_DAYS = 5
    TOP_PCT = 0.05
    STOP_LOSS = -0.08
    
    # 交易成本
    COST_RATE = 0.0013  # 0.13%单次
    
    dates = sorted(df['date'].unique())
    trades = []
    positions = {}  # symbol -> {buy_date, buy_price, max_price}
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1]
        
        today_df = df[df['date'] == today]
        tomorrow_df = df[df['date'] == tomorrow]
        
        # 1. 检查止损
        for symbol, pos in list(positions.items()):
            stock_today = today_df[today_df['symbol'] == symbol]
            if len(stock_today) == 0:
                continue
            
            current_price = stock_today['close'].values[0]
            buy_price = pos['buy_price']
            current_return = (current_price / buy_price) - 1
            
            if current_price > pos.get('max_price', buy_price):
                pos['max_price'] = current_price
            
            if current_return <= STOP_LOSS:
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * (1 - COST_RATE)
                    ret = (sell_price / buy_price) - 1
                    stock_name = stock_tomorrow['stock_name'].values[0]
                    
                    trades.append({
                        'trade_date': tomorrow.strftime('%Y-%m-%d'),
                        'signal_date': today.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'stock_name': stock_name,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': (tomorrow - pos['buy_date']).days,
                        'buy_signal': f'预测得分{pos["pred"]:.6f}排名前{TOP_PCT*100:.0f}%',
                        'sell_signal': '-8%止损',
                        'pred_score': pos['pred']
                    })
                    del positions[symbol]
        
        # 2. 检查持仓到期
        for symbol, pos in list(positions.items()):
            if (today - pos['buy_date']).days >= HOLDING_DAYS:
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * (1 - COST_RATE)
                    ret = (sell_price / pos['buy_price']) - 1
                    stock_name = stock_tomorrow['stock_name'].values[0]
                    
                    trades.append({
                        'trade_date': tomorrow.strftime('%Y-%m-%d'),
                        'signal_date': today.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'stock_name': stock_name,
                        'buy_price': pos['buy_price'],
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': HOLDING_DAYS,
                        'buy_signal': f'预测得分{pos["pred"]:.6f}排名前{TOP_PCT*100:.0f}%',
                        'sell_signal': '持仓到期',
                        'pred_score': pos['pred']
                    })
                    del positions[symbol]
        
        # 3. 买入新股票
        n_select = max(1, int(len(today_df) * TOP_PCT))
        available = today_df[~today_df['symbol'].isin(positions.keys())]
        
        if len(available) >= n_select:
            top_stocks = available.nlargest(n_select, 'pred')
            
            for _, stock in top_stocks.iterrows():
                symbol = stock['symbol']
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                
                if len(stock_tomorrow) > 0:
                    buy_price = stock_tomorrow['open'].values[0] * (1 + COST_RATE)
                    
                    positions[symbol] = {
                        'buy_date': tomorrow,
                        'buy_price': buy_price,
                        'max_price': buy_price,
                        'pred': stock['pred']
                    }
    
    # 处理最后持仓
    if len(dates) > 0 and len(positions) > 0:
        last_date = dates[-1]
        last_df = df[df['date'] == last_date]
        
        for symbol, pos in positions.items():
            stock_last = last_df[last_df['symbol'] == symbol]
            if len(stock_last) > 0:
                sell_price = stock_last['close'].values[0] * (1 - COST_RATE)
                ret = (sell_price / pos['buy_price']) - 1
                stock_name = stock_last['stock_name'].values[0]
                
                trades.append({
                    'trade_date': last_date.strftime('%Y-%m-%d'),
                    'signal_date': (last_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'stock_name': stock_name,
                    'buy_price': pos['buy_price'],
                    'sell_price': sell_price,
                    'return': ret,
                    'holding_days': (last_date - pos['buy_date']).days,
                    'buy_signal': f'预测得分{pos["pred"]:.6f}排名前{TOP_PCT*100:.0f}%',
                    'sell_signal': '期末清仓',
                    'pred_score': pos['pred']
                })
    
    # 分析结果
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ 无交易数据")
        return
    
    # 统计
    avg_return = trades_df['return'].mean()
    win_rate = (trades_df['return'] > 0).mean()
    weekly_return = avg_return
    annual_return = (1 + weekly_return) ** 52 - 1
    annual_cost = COST_RATE * 2 * 52
    
    print(f"\n{'='*70}")
    print("📊 V2模型回测结果")
    print(f"{'='*70}")
    print(f"总交易: {len(trades_df)} 笔")
    print(f"平均收益: {avg_return*100:.3f}%")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"年化收益(毛): {annual_return*100:.2f}%")
    print(f"年化成本: {annual_cost*100:.2f}%")
    print(f"年化收益(净): {(annual_return-annual_cost)*100:.2f}%")
    print(f"平均持仓: {trades_df['holding_days'].mean():.1f} 天")
    print(f"\n卖出原因:")
    print(f"  止损(-8%): {(trades_df['sell_signal']=='-8%止损').sum()} 笔")
    print(f"  持仓到期: {(trades_df['sell_signal']=='持仓到期').sum()} 笔")
    print(f"  期末清仓: {(trades_df['sell_signal']=='期末清仓').sum()} 笔")
    
    # 保存到result文件夹
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    
    # 保存交易明细（包含股票名称）
    output_file = result_dir / "backtest_v2_trades.csv"
    trades_df.to_csv(output_file, index=False)
    print(f"\n✅ 交易明细保存: {output_file}")
    
    # 保存汇总
    summary = {
        'model_version': 'v2',
        'model_ic': float(model_data.get('ensemble_ic', 0)),
        'total_trades': len(trades_df),
        'avg_return': float(avg_return),
        'win_rate': float(win_rate),
        'annual_return_gross': float(annual_return),
        'annual_cost': float(annual_cost),
        'annual_return_net': float(annual_return - annual_cost),
        'avg_holding_days': float(trades_df['holding_days'].mean()),
        'data_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        }
    }
    
    with open(result_dir / 'backtest_v2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ 汇总保存: {result_dir}/backtest_v2_summary.json")
    
    # 显示前10笔交易
    print(f"\n{'='*70}")
    print("📋 前10笔交易明细")
    print(f"{'='*70}")
    display_cols = ['trade_date', 'symbol', 'stock_name', 'buy_price', 'sell_price', 'return', 'sell_signal']
    print(trades_df[display_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    backtest_v2_final()
