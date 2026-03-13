#!/usr/bin/env python3
"""
为 CSI300、CSI500、CSI1000 生成特征数据
"""

import sys
import pandas as pd
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workflow.feature_engineering import FeatureEngineer


def load_stock_data_from_cache(symbol, cache_dir):
    """从缓存加载单只股票数据"""
    cache_file = cache_dir / f"daily_{symbol}_2020-01-01_2025-08-31.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
            # 确保使用 ts_code 作为 symbol
            if 'ts_code' in df.columns:
                df['symbol'] = df['ts_code']
            return df
    return None


def prepare_index_data(index_name, stock_list_file, cache_dir, output_dir, train_end='2024-08-31', valid_start='2024-09-01'):
    """为指定指数准备特征数据"""
    print(f"\n{'='*60}")
    print(f"🔧 处理 {index_name.upper()} 数据")
    print(f"{'='*60}")
    
    # 加载股票列表
    with open(stock_list_file, 'rb') as f:
        stocks = pickle.load(f)
    print(f"📊 {index_name} 成分股: {len(stocks)} 只")
    
    # 加载所有股票数据
    all_data = []
    missing_stocks = []
    
    for symbol in stocks:
        df = load_stock_data_from_cache(symbol, cache_dir)
        if df is not None and not df.empty:
            all_data.append(df)
        else:
            missing_stocks.append(symbol)
    
    if missing_stocks:
        print(f"⚠️ 缺失数据的股票: {len(missing_stocks)} 只")
    
    if not all_data:
        print(f"❌ {index_name} 没有可用数据")
        return False
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 加载数据: {len(combined_df)} 条记录, {combined_df['symbol'].nunique()} 只股票")
    
    # 转换日期格式
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # 分割训练集和验证集
    train_df = combined_df[combined_df['date'] <= train_end].copy()
    valid_df = combined_df[combined_df['date'] >= valid_start].copy()
    
    print(f"\n📅 训练集: {train_df['date'].min().date()} ~ {train_df['date'].max().date()} ({len(train_df)} 条)")
    print(f"📅 验证集: {valid_df['date'].min().date()} ~ {valid_df['date'].max().date()} ({len(valid_df)} 条)")
    
    # 特征工程
    engineer = FeatureEngineer()
    
    print(f"\n🔧 计算训练集特征...")
    train_features = engineer.calculate_all_features(train_df)
    
    print(f"🔧 计算验证集特征...")
    valid_features = engineer.calculate_all_features(valid_df)
    
    # 保存数据
    index_dir = output_dir / index_name
    index_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = index_dir / "train_features.csv"
    valid_file = index_dir / "valid_features.csv"
    
    train_features.to_csv(train_file, index=False)
    valid_features.to_csv(valid_file, index=False)
    
    print(f"\n✅ {index_name.upper()} 特征数据保存完成:")
    print(f"   训练集: {train_file} ({train_features.shape})")
    print(f"   验证集: {valid_file} ({valid_features.shape})")
    
    return True


def main():
    """主函数"""
    print("="*60)
    print("🚀 为三大指数生成特征数据")
    print("="*60)
    
    cache_dir = Path("data/cache")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    indices = [
        ('csi300', cache_dir / 'stock_list_csi300.pkl'),
        ('csi1000', cache_dir / 'stock_list_csi1000.pkl'),
    ]
    
    results = {}
    for index_name, stock_list_file in indices:
        if stock_list_file.exists():
            success = prepare_index_data(index_name, stock_list_file, cache_dir, output_dir)
            results[index_name] = success
        else:
            print(f"❌ 找不到股票列表: {stock_list_file}")
            results[index_name] = False
    
    # 汇总
    print(f"\n{'='*60}")
    print("📊 处理完成汇总")
    print(f"{'='*60}")
    
    for index_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{index_name.upper()}: {status}")
    
    print(f"\n{'='*60}")
    print("✅ 所有处理完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
