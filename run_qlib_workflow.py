#!/usr/bin/env python3
"""
Qlib 风格工作流主脚本

运行：
    python run_qlib_workflow.py
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from qlib_workflow.data import QlibData
from qlib_workflow.features import QlibFeatures
from qlib_workflow.model import QlibModel
from qlib_workflow.strategy import QlibStrategy
from qlib_workflow.backtest import QlibBacktest

from tushare_provider.tushare_fetcher import TushareDataFetcher
from workflow.feature_engineering_v7 import V7FeatureEngineer


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Qlib 风格量化工作流")
    print("=" * 60)
    
    # 加载配置
    with open("config/workflow_config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # ========== Step 1: Data ==========
    print("\n📥 Step 1: 数据加载")
    fetcher = TushareDataFetcher(
        cfg['data']['tushare_token'],
        cfg['data']['cache_path']
    )
    
    stocks = fetcher.get_stock_list(cfg['data']['market'])
    print(f"✅ 股票数量: {len(stocks)}")
    
    price_df = fetcher.get_daily_prices_batch(
        stocks[:100],  # 先测试100只
        cfg['data']['train_start'],
        cfg['data']['valid_end']
    )
    print(f"✅ 记录数: {len(price_df)}")
    
    if price_df.empty:
        print("❌ 数据为空")
        return
    
    # ========== Step 2: Features ==========
    print("\n🔧 Step 2: 特征工程")
    
    # 使用 V7 特征
    df = V7FeatureEngineer.calculate_all_features(price_df)
    features = V7FeatureEngineer.get_feature_cols()
    features = [c for c in features if c in df.columns]
    
    # 或使用 Qlib 风格特征
    # qlib_features = QlibFeatures()
    # df = qlib_features.calc_alpha_features(price_df)
    # features = qlib_features.get_feature_names()
    
    print(f"✅ 特征数: {len(features)}")
    
    # ========== Step 3: Model ==========
    print("\n🤖 Step 3: 模型训练")
    
    # 划分数据集
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
    model = QlibModel(model_type='lightgbm')
    model.fit((X_train, y_train, X_valid, y_valid))
    
    # 评估
    ic = model.score(X_valid, y_valid)
    print(f"✅ 验证集 IC: {ic:.4f}")
    
    # 保存模型
    model.save(f"{cfg['output']['model_dir']}/qlib_model.pkl")
    
    # ========== Step 4: Strategy ==========
    print("\n📈 Step 4: 策略生成")
    
    # 预测
    valid_df['score'] = model.predict(X_valid)
    
    # 生成信号
    strategy = QlibStrategy(model, topk=50, drop=0)
    signals = strategy.generate_signals(valid_df, features)
    
    # 获取最新选股
    latest_date = valid_df['date'].max()
    selection = strategy.get_daily_selection(signals, latest_date)
    
    print(f"✅ 选股数量: {len(selection)}")
    print(f"\nTop 10:")
    print(selection.head(10))
    
    # 保存选股
    selection.to_csv(f"{cfg['output']['selection_dir']}/selection_{latest_date}.csv")
    
    # ========== Step 5: Backtest ==========
    print("\n📊 Step 5: 回测")
    
    backtest = QlibBacktest(
        cfg['data']['valid_start'],
        cfg['data']['valid_end']
    )
    
    # 简化回测
    # metrics = backtest.run(signals, price_df)
    # backtest.report()
    
    print("\n" + "=" * 60)
    print("✅ Qlib 工作流完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
