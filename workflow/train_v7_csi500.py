#!/usr/bin/env python3
"""
V7 模型训练主脚本 - CSI500

运行：
    python workflow/train_v7_csi500.py
"""

import sys
import yaml
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tushare_provider.tushare_fetcher import TushareDataFetcher
from workflow.feature_engineering_v7 import V7FeatureEngineer
from workflow.model_ensemble_v7 import V7Trainer


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 V7 模型训练 - CSI500")
    print("=" * 60)
    
    # 加载配置
    with open("config/workflow_config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # 初始化数据获取器
    fetcher = TushareDataFetcher(
        cfg['data']['tushare_token'],
        cfg['data']['cache_path']
    )
    
    # 获取股票列表
    print("\n📥 获取 CSI500 成分股...")
    stocks = fetcher.get_stock_list('csi500')
    print(f"✅ 股票数量: {len(stocks)}")
    
    # 获取价格数据
    print("\n📥 获取价格数据...")
    price_df = fetcher.get_daily_prices_batch(
        stocks,
        cfg['data']['train_start'],
        cfg['data']['valid_end']
    )
    print(f"✅ 记录数: {len(price_df)}")
    
    if price_df.empty:
        print("❌ 数据为空，退出")
        return
    
    # 特征工程
    print("\n🔧 特征工程...")
    df = V7FeatureEngineer.calculate_all_features(price_df)
    
    features = V7FeatureEngineer.get_feature_cols()
    features = [c for c in features if c in df.columns]
    print(f"✅ 特征数: {len(features)}")
    
    # 划分数据集
    print("\n📊 划分数据集...")
    train_df = df[
        (df['date'] >= cfg['data']['train_start']) &
        (df['date'] <= cfg['data']['train_end'])
    ].dropna(subset=features + ['label'])
    
    valid_df = df[
        (df['date'] >= cfg['data']['valid_start']) &
        (df['date'] <= cfg['data']['valid_end'])
    ].dropna(subset=features + ['label'])
    
    X_train, y_train = train_df[features], train_df['label']
    X_valid, y_valid = valid_df[features], valid_df['label']
    
    print(f"✅ 训练集: {len(X_train)}, 验证集: {len(X_valid)}")
    
    # 训练模型
    print("\n🤖 训练模型...")
    trainer = V7Trainer('csi500')
    eic = trainer.train(X_train, y_train, X_valid, y_valid)
    
    # 保存模型
    model_path = f"{cfg['output']['model_dir']}/model_enhanced_v7_csi500.pkl"
    trainer.save(model_path, features, eic)
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成! 集成IC: {eic:.4f}")
    print(f"📁 模型保存: {model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
