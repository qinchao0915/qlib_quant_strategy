#!/usr/bin/env python3
"""
回测模型 V2 - 改进版
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


def calculate_market_trend(index_df, current_date):
    """
    计算大盘趋势
    返回: 1.0(满仓), 0.5(半仓), 0.0(空仓)
    """
    # 获取当前日期的指数数据
    idx_data = index_df[index_df['date'] <= current_date].tail(60)
    if len(idx_data) < 60:
        return 1.0  # 数据不足，默认满仓
    
    close = idx_data['close'].values
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])
    current = close[-1]
    
    # 大盘趋势判断
    if current > ma20 and ma20 > ma60:
        return 1.0  # 多头排列，满仓
    elif current > ma60:
        return 0.5  # 价格在60日线上方，半仓
    else:
        return 0.0  # 空头，空仓


def calculate_transaction_cost(price, is_buy=True):
    """
    计算交易成本
    买入：价格上浮（滑点）
    卖出：价格下浮（滑点）
    """
    # 手续费：万分之3（买卖双向）
    commission_rate = 0.0003
    # 滑点：0.1%
    slippage = 0.001
    
    # 单次交易成本 = 手续费 + 滑点
    total_cost_rate = commission_rate + slippage
    
    if is_buy:
        # 买入：实际成本 = 价格 * (1 + 总成本率)
        effective_price = price * (1 + total_cost_rate)
    else:
        # 卖出：实际收入 = 价格 * (1 - 总成本率)
        effective_price = price * (1 - total_cost_rate)
    
    return effective_price


def backtest_index_v2(index_name, model_path, data_dir, 
                      start_date='2025-01-01', end_date='2025-12-31',
                      holding_days=5, top_pct=0.05):
    """
    回测单个指数 - V2改进版
    
    Args:
        holding_days: 最少持仓天数
        top_pct: 选股比例（默认5%）
    """
    print(f"\n{'='*80}")
    print(f"📊 回测 {index_name.upper()} - V2改进版")
    print(f"{'='*80}")
    print(f"参数: 持仓{holding_days}天, 选股前{top_pct*100}%")
    
    # 加载模型
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model_data = load_model(model_path)
    features = model_data['features']
    
    # 加载股票数据
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
    
    # 加载大盘指数数据（用于趋势判断）
    # 使用指数成分股的平均走势作为大盘代理
    index_proxy = df.groupby('date').agg({
        'close': 'mean',
        'open': 'mean',
        'high': 'mean',
        'low': 'mean'
    }).reset_index()
    index_proxy.columns = ['date', 'close', 'open', 'high', 'low']
    
    print(f"✅ 加载数据: {len(df)} 条记录, {df['date'].nunique()} 个交易日")
    
    # 确保所有特征都存在
    features = [f for f in features if f in df.columns]
    
    # 预测
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 初始化持仓
    positions = {}  # symbol -> {buy_date, buy_price, holding_days}
    trades = []
    daily_returns = []
    
    dates = sorted(df['date'].unique())
    
    for i, current_date in enumerate(dates):
        # 获取当日数据
        current_df = df[df['date'] == current_date].copy()
        
        # 获取次日数据（用于计算收益）
        future_dates = dates[i+1:i+holding_days+1]
        if len(future_dates) < holding_days:
            continue  # 剩余天数不足，跳过
        
        # 计算大盘趋势
        market_position = calculate_market_trend(index_proxy, current_date)
        
        if market_position == 0:
            # 空仓，卖出所有持仓
            for symbol, pos in list(positions.items()):
                stock_data = current_df[current_df['symbol'] == symbol]
                if len(stock_data) > 0:
                    sell_price = calculate_transaction_cost(
                        stock_data['close'].values[0], is_buy=False
                    )
                    buy_price = pos['buy_price']
                    ret = (sell_price / buy_price) - 1
                    
                    trades.append({
                        'trade_date': current_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': pos['holding_days'],
                        'reason': 'market_filter_exit'
                    })
                    del positions[symbol]
            continue
        
        # 检查持仓是否需要卖出
        for symbol, pos in list(positions.items()):
            pos['holding_days'] += 1
            
            # 卖出条件：1) 持有满holding_days天 2) 当日有数据
            if pos['holding_days'] >= holding_days:
                stock_data = current_df[current_df['symbol'] == symbol]
                if len(stock_data) > 0:
                    # 计算卖出价格（扣除成本）
                    sell_price = calculate_transaction_cost(
                        stock_data['close'].values[0], is_buy=False
                    )
                    buy_price = pos['buy_price']
                    ret = (sell_price / buy_price) - 1
                    
                    trades.append({
                        'trade_date': current_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'return': ret,
                        'holding_days': pos['holding_days'],
                        'reason': 'holding_period_end'
                    })
                    del positions[symbol]
        
        # 买入新股票
        # 计算可买入数量
        n_select = max(1, int(len(current_df) * top_pct * market_position))
        
        if n_select > 0 and len(positions) < n_select * 2:  # 允许最多2倍持仓
            # 按预测值排序
            top_stocks = current_df.nlargest(n_select, 'pred')
            
            for _, stock in top_stocks.iterrows():
                symbol = stock['symbol']
                if symbol not in positions:
                    # 计算买入价格（加上成本）
                    buy_price = calculate_transaction_cost(
                        stock['open'], is_buy=True
                    )
                    
                    positions[symbol] = {
                        'buy_date': current_date,
                        'buy_price': buy_price,
                        'holding_days': 0,
                        'pred_score': stock['pred']
                    }
        
        # 计算当日组合收益（基于持仓市值变化）
        if len(positions) > 0:
            position_returns = []
            for symbol, pos in positions.items():
                stock_data = current_df[current_df['symbol'] == symbol]
                if len(stock_data) > 0:
                    current_price = stock_data['close'].values[0]
                    ret = (current_price / pos['buy_price']) - 1
                    position_returns.append(ret)
            
            daily_return = np.mean(position_returns) if position_returns else 0
        else:
            daily_return = 0
        
        daily_returns.append({
            'date': current_date,
            'return': daily_return,
            'n_positions': len(positions),
            'market_position': market_position
        })
    
    # 处理最后未卖出的持仓
    last_date = dates[-1]
    last_df = df[df['date'] == last_date]
    for symbol, pos in positions.items():
        stock_data = last_df[last_df['symbol'] == symbol]
        if len(stock_data) > 0:
            sell_price = calculate_transaction_cost(
                stock_data['close'].values[0], is_buy=False
            )
            buy_price = pos['buy_price']
            ret = (sell_price / buy_price) - 1
            
            trades.append({
                'trade_date': last_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': ret,
                'holding_days': pos['holding_days'],
                'reason': 'end_of_period'
            })
    
    # 转换为DataFrame
    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_returns)
    
    if len(trades_df) == 0 or len(daily_df) == 0:
        print("❌ 没有生成有效的交易数据")
        return None
    
    # 计算累计收益
    daily_df = daily_df.sort_values('date')
    daily_df['cum_return'] = (1 + daily_df['return']).cumprod() - 1
    
    # 计算指标
    total_return = daily_df['cum_return'].iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(daily_df)) - 1
    volatility = daily_df['return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = (daily_df['cum_return'] - daily_df['cum_return'].cummax()).min()
    win_rate = (daily_df['return'] > 0).mean()
    
    # 计算交易成本
    total_trades = len(trades_df)
    total_cost = total_trades * 0.0013 * 2  # 单次成本0.13%，买卖双向
    
    print(f"\n{'='*80}")
    print(f"📊 {index_name.upper()} V2回测结果")
    print(f"{'='*80}")
    print(f"总收益率:     {total_return*100:>8.2f}%")
    print(f"年化收益率:   {annual_return*100:>8.2f}%")
    print(f"年化波动率:   {volatility*100:>8.2f}%")
    print(f"夏普比率:     {sharpe_ratio:>8.2f}")
    print(f"最大回撤:     {max_drawdown*100:>8.2f}%")
    print(f"日胜率:       {win_rate*100:>8.2f}%")
    print(f"交易天数:     {len(daily_df):>8} 天")
    print(f"总交易次数:   {total_trades:>8} 笔")
    print(f"预估总成本:   {total_cost*100:>8.2f}%")
    print(f"净收益率:     {(total_return-total_cost)*100:>8.2f}%")
    
    # 保存结果
    output_dir = Path("backtest_result")
    output_dir.mkdir(exist_ok=True)
    
    trades_df.to_csv(output_dir / f"trades_{index_name}_2025_v2.csv", index=False)
    daily_df.to_csv(output_dir / f"daily_{index_name}_2025_v2.csv", index=False)
    
    print(f"\n✅ 结果保存:")
    print(f"   交易明细: {output_dir}/trades_{index_name}_2025_v2.csv")
    print(f"   每日收益: {output_dir}/daily_{index_name}_2025_v2.csv")
    
    return trades_df, daily_df


def main():
    """主函数"""
    print("="*80)
    print("🚀 V2改进版回测")
    print("改进点: 交易成本 + 大盘过滤 + 5天持仓")
    print("="*80)
    
    data_dir = Path("data/processed")
    model_dir = Path("model")
    
    # 回测CSI500
    index_name = 'csi500'
    model_path = model_dir / f"model_{index_name}.pkl"
    
    result = backtest_index_v2(
        index_name, 
        model_path, 
        data_dir,
        holding_days=5,  # 持仓5天
        top_pct=0.05     # 选股前5%
    )
    
    if result:
        trades_df, daily_df = result
        
        print(f"\n{'='*80}")
        print("📋 交易统计")
        print(f"{'='*80}")
        print(f"平均持仓天数: {trades_df['holding_days'].mean():.1f} 天")
        print(f"平均单笔收益: {trades_df['return'].mean()*100:.3f}%")
        print(f"盈利交易: {(trades_df['return'] > 0).sum()} 笔")
        print(f"亏损交易: {(trades_df['return'] < 0).sum()} 笔")
        
        # 大盘过滤统计
        market_stats = daily_df.groupby('market_position')['return'].agg(['count', 'mean'])
        print(f"\n📊 大盘过滤统计:")
        for pos, stats in market_stats.iterrows():
            pos_name = {0.0: '空仓', 0.5: '半仓', 1.0: '满仓'}.get(pos, '未知')
            print(f"   {pos_name}: {stats['count']} 天, 日均收益 {stats['mean']*100:.3f}%")
    
    print(f"\n{'='*80}")
    print("✅ V2回测完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
