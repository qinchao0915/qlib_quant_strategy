#!/usr/bin/env python3
"""
V3优化版回测
- 修复持仓天数逻辑（严格>=5天）
- 大盘风控（MA20/MA60择时）
- 移动止损（最高价回撤8%）
- 换手惩罚（前15%保留）
- 使用V2模型
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
    name_file = Path('data/cache/stock_name_map_full.json')
    if not name_file.exists():
        name_file = Path('data/cache/stock_name_map.json')
    
    if name_file.exists():
        with open(name_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def predict(model_data, X):
    """模型预测"""
    models = model_data['models']
    weights = model_data['weights']
    predictions = []
    for name, model in models.items():
        if name in weights and weights[name] > 0:
            pred = model.predict(X)
            predictions.append(weights[name] * pred)
    return np.sum(predictions, axis=0)


def calculate_market_trend(df, current_date):
    """
    计算大盘趋势（MA20/MA60）
    返回: True(可以交易) / False(停止开仓)
    """
    # 获取指数数据（成分股平均）
    idx_data = df[df['date'] <= current_date].groupby('date')['close'].mean().reset_index()
    idx_data = idx_data.tail(60)
    
    if len(idx_data) < 60:
        return True  # 数据不足，允许交易
    
    close = idx_data['close'].values
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])
    current = close[-1]
    
    # 如果收盘价同时低于MA20和MA60，停止开仓
    if current < ma20 and current < ma60:
        return False  # 空头，停止开仓
    return True  # 可以交易


def backtest_v3_optimized():
    """V3优化版回测"""
    print("="*80)
    print("🚀 V3优化版回测")
    print("改进: 严格持仓5天 + 大盘择时 + 移动止损 + 换手惩罚")
    print("="*80)
    
    # 加载V2模型
    model_path = Path("model/model_csi500_v2.pkl")
    model_data = load_model(model_path)
    features = model_data['features']
    
    print(f"✅ 加载V2模型 (IC: {model_data.get('ensemble_ic', 'N/A'):.4f})")
    
    # 加载股票名称
    stock_names = load_stock_names()
    
    # 加载数据
    valid_file = Path("data/processed/valid_features.csv")
    df = pd.read_csv(valid_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2025-01-01']
    
    # 添加股票名称
    df['stock_name'] = df['symbol'].map(stock_names)
    df['stock_name'] = df['stock_name'].fillna('未知')
    
    # 预测
    features = [f for f in features if f in df.columns]
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 参数
    HOLDING_DAYS = 5
    TOP_PCT_NEW = 0.05      # 新买入选股比例
    TOP_PCT_KEEP = 0.15     # 保留持仓阈值
    TRAILING_STOP = -0.08   # 移动止损8%
    COST_RATE = 0.0013      # 交易成本
    
    dates = sorted(df['date'].unique())
    trades = []
    
    # 持仓结构: symbol -> {buy_date, buy_price, max_price, pred}
    positions = {}
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1]
        
        today_df = df[df['date'] == today]
        tomorrow_df = df[df['date'] == tomorrow]
        
        # ===== 1. 更新持仓最高价（用于移动止损）=====
        for symbol, pos in positions.items():
            stock_today = today_df[today_df['symbol'] == symbol]
            if len(stock_today) > 0:
                current_price = stock_today['close'].values[0]
                if current_price > pos['max_price']:
                    pos['max_price'] = current_price
        
        # ===== 2. 检查移动止损 =====
        for symbol, pos in list(positions.items()):
            stock_today = today_df[today_df['symbol'] == symbol]
            if len(stock_today) == 0:
                continue
            
            current_price = stock_today['close'].values[0]
            max_price = pos['max_price']
            
            # 移动止损：从最高价回撤8%
            drawdown = (current_price - max_price) / max_price
            
            if drawdown <= TRAILING_STOP:
                # 明天开盘卖出
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
                        'holding_days': (tomorrow - pos['buy_date']).days,
                        'buy_signal': f'预测得分{pos["pred"]:.6f}',
                        'sell_signal': f'移动止损({drawdown*100:.1f}%)',
                        'pred_score': pos['pred']
                    })
                    del positions[symbol]
        
        # ===== 3. 大盘择时：判断是否可以开仓 =====
        can_open = calculate_market_trend(df, today)
        
        if not can_open:
            print(f"  {today.strftime('%Y-%m-%d')}: 大盘空头，停止开仓")
        
        # ===== 4. 持仓到期与换手惩罚（合并逻辑）=====
        # 获取今日所有股票的预测得分排名
        today_df_sorted = today_df.sort_values('pred', ascending=False)
        n_total = len(today_df_sorted)
        
        # 保留持仓阈值：当前得分在前15%
        keep_threshold_idx = int(n_total * TOP_PCT_KEEP)
        keep_threshold_score = today_df_sorted.iloc[keep_threshold_idx]['pred'] if keep_threshold_idx < n_total else today_df_sorted['pred'].min()
        
        # 检查持仓是否到期或需要换仓
        for symbol, pos in list(positions.items()):
            # 如果已经触发止损，上面已经处理了
            if symbol not in positions:
                continue
            
            stock_today = today_df[today_df['symbol'] == symbol]
            if len(stock_today) == 0:
                continue
            
            current_pred = stock_today['pred'].values[0]
            days_held = (today - pos['buy_date']).days
            
            # 只有持仓满5天才检查是否卖出
            if days_held >= HOLDING_DAYS:
                stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                if len(stock_tomorrow) > 0:
                    sell_price = stock_tomorrow['open'].values[0] * (1 - COST_RATE)
                    ret = (sell_price / pos['buy_price']) - 1
                    stock_name = stock_tomorrow['stock_name'].values[0]
                    
                    # 判断卖出原因：排名在前15%则继续持有，否则换仓
                    if current_pred >= keep_threshold_score:
                        # 排名在前15%，继续持有（不卖出）
                        continue
                    else:
                        # 排名跌出前15%，触发换仓
                        trades.append({
                            'trade_date': tomorrow.strftime('%Y-%m-%d'),
                            'signal_date': today.strftime('%Y-%m-%d'),
                            'symbol': symbol,
                            'stock_name': stock_name,
                            'buy_price': pos['buy_price'],
                            'sell_price': sell_price,
                            'return': ret,
                            'holding_days': days_held,
                            'buy_signal': f'预测得分{pos["pred"]:.6f}',
                            'sell_signal': '排名下降(换仓)',
                            'pred_score': pos['pred']
                        })
                        del positions[symbol]
        
        # ===== 6. 买入新股票（仅在允许开仓时）=====
        if can_open:
            # 计算需要买入的数量
            n_select = max(1, int(n_total * TOP_PCT_NEW))
            
            # 排除已有持仓的股票
            available = today_df_sorted[~today_df_sorted['symbol'].isin(positions.keys())]
            
            # 选前5%买入
            if len(available) >= n_select:
                top_stocks = available.head(n_select)
                
                for _, stock in top_stocks.iterrows():
                    symbol = stock['symbol']
                    stock_tomorrow = tomorrow_df[tomorrow_df['symbol'] == symbol]
                    
                    if len(stock_tomorrow) > 0:
                        buy_price = stock_tomorrow['open'].values[0] * (1 + COST_RATE)
                        stock_name = stock_tomorrow['stock_name'].values[0]
                        
                        positions[symbol] = {
                            'buy_date': tomorrow,
                            'buy_price': buy_price,
                            'max_price': buy_price,
                            'pred': stock['pred']
                        }
    
    # 处理最后未卖出的持仓
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
                    'buy_signal': f'预测得分{pos["pred"]:.6f}',
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
    
    print(f"\n{'='*80}")
    print("📊 V3优化版回测结果")
    print(f"{'='*80}")
    print(f"总交易: {len(trades_df)} 笔")
    print(f"平均收益: {avg_return*100:.3f}%")
    print(f"胜率: {win_rate*100:.1f}%")
    print(f"年化收益(毛): {annual_return*100:.2f}%")
    print(f"年化成本: {annual_cost*100:.2f}%")
    print(f"年化收益(净): {(annual_return-annual_cost)*100:.2f}%")
    print(f"平均持仓: {trades_df['holding_days'].mean():.1f} 天")
    
    # 卖出原因统计
    print(f"\n卖出原因分布:")
    for reason in trades_df['sell_signal'].unique():
        count = (trades_df['sell_signal'] == reason).sum()
        avg_ret = trades_df[trades_df['sell_signal'] == reason]['return'].mean()
        print(f"  {reason}: {count} 笔 (平均{avg_ret*100:.2f}%)")
    
    # 保存结果
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    trades_df.to_csv(result_dir / "backtest_v3_optimized.csv", index=False)
    
    # 保存汇总
    summary = {
        'version': 'v3_optimized',
        'model': 'v2',
        'total_trades': len(trades_df),
        'avg_return': float(avg_return),
        'win_rate': float(win_rate),
        'annual_return_gross': float(annual_return),
        'annual_cost': float(annual_cost),
        'annual_return_net': float(annual_return - annual_cost),
        'avg_holding_days': float(trades_df['holding_days'].mean()),
        'features': ['strict_holding_5d', 'ma20_ma60_filter', 'trailing_stop_8pct', 'top15_keep']
    }
    
    import json
    with open(result_dir / 'backtest_v3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 结果保存:")
    print(f"   交易明细: {result_dir}/backtest_v3_optimized.csv")
    print(f"   汇总: {result_dir}/backtest_v3_summary.json")
    
    # 显示前10笔
    print(f"\n{'='*80}")
    print("📋 前10笔交易")
    print(f"{'='*80}")
    display_cols = ['trade_date', 'symbol', 'stock_name', 'return', 'holding_days', 'sell_signal']
    print(trades_df[display_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    backtest_v3_optimized()
