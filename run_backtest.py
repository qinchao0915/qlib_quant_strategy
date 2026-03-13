#!/usr/bin/env python3
"""
回测模型 - 2025年全年收益率
"""

import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from workflow.feature_engineering import FeatureEngineer


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


def backtest_index(index_name, model_path, data_dir, start_date='2025-01-01', end_date='2025-12-31'):
    """回测单个指数"""
    print(f"\n{'='*60}")
    print(f"📊 回测 {index_name.upper()} - 2025年")
    print(f"{'='*60}")
    
    # 加载模型
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model_data = load_model(model_path)
    features = model_data['features']
    print(f"✅ 加载模型: {model_path}")
    print(f"   模型IC: {model_data.get('ensemble_ic', 'N/A'):.4f}")
    print(f"   特征数: {len(features)}")
    
    # 加载2025年数据
    if index_name == 'csi500':
        # CSI500 数据在根目录
        data_file = data_dir / "valid_features.csv"
    else:
        data_file = data_dir / index_name / "valid_features.csv"
    
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # 筛选2025年数据
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if df.empty:
        print(f"❌ 没有找到 {start_date} 到 {end_date} 的数据")
        return None
    
    print(f"✅ 加载数据: {len(df)} 条记录, {df['date'].nunique()} 个交易日")
    print(f"   股票数: {df['symbol'].nunique()}")
    print(f"   日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 确保所有特征都存在
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"⚠️ 缺失特征: {missing_features}")
        # 只使用存在的特征
        features = [f for f in features if f in df.columns]
    
    # 预测
    print(f"\n🔮 生成预测...")
    X = df[features].values
    df['pred'] = predict(model_data, X)
    
    # 计算每日收益率
    print(f"\n📈 计算收益率...")
    
    # 按日期分组，计算每日收益
    daily_returns = []
    
    for date, group in df.groupby('date'):
        # 按预测值排序，选择前20%的股票
        n_select = max(1, int(len(group) * 0.2))
        top_stocks = group.nlargest(n_select, 'pred')
        
        # 计算选中股票的平均收益率
        # 使用次日收益率作为标签
        avg_return = top_stocks['close'].mean() / top_stocks['pre_close'].mean() - 1
        
        daily_returns.append({
            'date': date,
            'return': avg_return,
            'n_stocks': n_select,
            'avg_pred': top_stocks['pred'].mean()
        })
    
    returns_df = pd.DataFrame(daily_returns)
    returns_df = returns_df.sort_values('date')
    
    # 计算累计收益率
    returns_df['cum_return'] = (1 + returns_df['return']).cumprod() - 1
    
    # 计算指标
    total_return = returns_df['cum_return'].iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(returns_df)) - 1
    volatility = returns_df['return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = (returns_df['cum_return'] - returns_df['cum_return'].cummax()).min()
    
    # 胜率
    win_rate = (returns_df['return'] > 0).mean()
    
    print(f"\n{'='*60}")
    print(f"📊 {index_name.upper()} 回测结果")
    print(f"{'='*60}")
    print(f"总收益率:     {total_return*100:>8.2f}%")
    print(f"年化收益率:   {annual_return*100:>8.2f}%")
    print(f"年化波动率:   {volatility*100:>8.2f}%")
    print(f"夏普比率:     {sharpe_ratio:>8.2f}")
    print(f"最大回撤:     {max_drawdown*100:>8.2f}%")
    print(f"日胜率:       {win_rate*100:>8.2f}%")
    print(f"交易天数:     {len(returns_df):>8} 天")
    
    return {
        'index': index_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'daily_returns': returns_df
    }


def main():
    """主函数"""
    print("="*60)
    print("🚀 2025年全年回测")
    print("="*60)
    
    data_dir = Path("data/processed")
    model_dir = Path("model")
    
    # 回测三个指数
    indices = ['csi300', 'csi500', 'csi1000']
    results = []
    
    for index_name in indices:
        model_path = model_dir / f"model_{index_name}.pkl"
        result = backtest_index(index_name, model_path, data_dir)
        if result:
            results.append(result)
    
    # 汇总
    if results:
        print(f"\n{'='*60}")
        print("📊 回测结果汇总")
        print(f"{'='*60}")
        print(f"{'指数':<10} {'总收益':<10} {'年化收益':<10} {'夏普':<8} {'最大回撤':<10} {'胜率':<8}")
        print("-"*60)
        
        for r in results:
            print(f"{r['index'].upper():<10} {r['total_return']*100:>8.2f}% {r['annual_return']*100:>8.2f}% {r['sharpe_ratio']:>6.2f} {r['max_drawdown']*100:>8.2f}% {r['win_rate']*100:>6.2f}%")
        
        print(f"{'='*60}")
        
        # 保存结果
        output_dir = Path("backtest_result")
        output_dir.mkdir(exist_ok=True)
        
        for r in results:
            output_file = output_dir / f"backtest_{r['index']}_2025.csv"
            r['daily_returns'].to_csv(output_file, index=False)
            print(f"✅ 详细结果保存: {output_file}")
    
    print(f"\n{'='*60}")
    print("✅ 回测完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
