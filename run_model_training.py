#!/usr/bin/env python3
"""
模型训练 - 使用已生成的特征数据
分别训练 CSI300、CSI500、CSI1000 模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from workflow.model_ensemble import V7Trainer as ModelTrainer


def train_model(market_name, train_file, valid_file):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"🚀 训练 {market_name} 模型")
    print(f"{'='*60}")
    
    # 读取数据
    print(f"\n📥 读取训练数据...")
    train_df = pd.read_csv(train_file)
    print(f"✅ 训练数据: {len(train_df)} 条")
    
    print(f"\n📥 读取验证数据...")
    valid_df = pd.read_csv(valid_file)
    print(f"✅ 验证数据: {len(valid_df)} 条")
    
    # 获取特征列（排除非特征列）
    exclude_cols = ['ts_code', 'date', 'open', 'high', 'low', 'close', 
                    'pre_close', 'change', 'pct_chg', 'volume', 'amount', 'symbol']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"\n📊 特征数量: {len(feature_cols)}")
    
    # 创建标签（次日收益率）
    print(f"\n🏷️ 创建标签...")
    train_df['label'] = train_df.groupby('symbol')['close'].shift(-1) / train_df['close'] - 1
    valid_df['label'] = valid_df.groupby('symbol')['close'].shift(-1) / valid_df['close'] - 1
    
    # 删除缺失值
    train_df = train_df.dropna(subset=feature_cols + ['label'])
    valid_df = valid_df.dropna(subset=feature_cols + ['label'])
    
    print(f"✅ 训练集: {len(train_df)}, 验证集: {len(valid_df)}")
    
    # 准备数据
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['label']
    
    # 训练模型
    print(f"\n🤖 训练 LightGBM 模型...")
    trainer = ModelTrainer(market_name)
    
    try:
        ic = trainer.train(X_train, y_train, X_valid, y_valid)
        
        # 保存模型
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"model_{market_name}.pkl"
        
        # 使用 V7Trainer 的 save 方法
        trainer.save(model_path, feature_cols, ic)
        
        print(f"\n{'='*60}")
        print(f"✅ {market_name} 模型训练完成!")
        print(f"📊 IC: {ic:.4f}")
        print(f"📁 模型保存: {model_path}")
        print(f"{'='*60}")
        
        return ic
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("="*60)
    print("🚀 开始训练三个指数模型")
    print("="*60)
    
    data_dir = Path("data/processed")
    results = {}
    
    # 定义三个指数的数据路径
    indices = [
        ('csi300', data_dir / 'csi300'),
        ('csi500', data_dir),  # CSI500 在根目录
        ('csi1000', data_dir / 'csi1000'),
    ]
    
    for market_name, market_dir in indices:
        train_file = market_dir / "train_features.csv"
        valid_file = market_dir / "valid_features.csv"
        
        if not train_file.exists():
            print(f"\n⚠️ 跳过 {market_name}: 找不到训练数据 {train_file}")
            continue
        
        # 训练模型
        ic = train_model(market_name, train_file, valid_file)
        results[market_name] = ic
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 训练结果汇总")
    print("="*60)
    
    for market_name, ic in results.items():
        if ic is not None:
            print(f"✅ {market_name.upper()}: IC = {ic:.4f}")
        else:
            print(f"❌ {market_name.upper()}: 训练失败")
    
    print("="*60)


if __name__ == '__main__':
    main()
