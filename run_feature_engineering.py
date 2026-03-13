#!/usr/bin/env python3
"""
运行特征工程 - 处理实际数据
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workflow.feature_engineering import FeatureEngineer


def main():
    """主函数"""
    print("=" * 60)
    print("🔧 特征工程 - 处理 CSI500 数据")
    print("=" * 60)
    
    # 读取原始数据
    train_file = Path("data/raw/csi500_train_2020-01-01_2024-08-31.csv")
    valid_file = Path("data/raw/csi500_valid_2024-09-01_2025-08-31.csv")
    
    if not train_file.exists():
        print(f"❌ 找不到训练数据: {train_file}")
        return
    
    print(f"\n📥 读取训练数据...")
    train_df = pd.read_csv(train_file)
    print(f"✅ 训练数据: {len(train_df)} 条记录, {len(train_df['symbol'].unique())} 只股票")
    
    # 初始化特征工程器
    engineer = FeatureEngineer()
    
    # 计算特征
    print(f"\n🔧 计算训练集特征...")
    train_features = engineer.calculate_all_features(train_df)
    
    # 保存特征
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / "train_features.csv"
    train_features.to_csv(train_output, index=False)
    print(f"✅ 训练特征保存: {train_output}")
    print(f"   形状: {train_features.shape}")
    
    # 处理验证集
    if valid_file.exists():
        print(f"\n📥 读取验证数据...")
        valid_df = pd.read_csv(valid_file)
        print(f"✅ 验证数据: {len(valid_df)} 条记录")
        
        print(f"\n🔧 计算验证集特征...")
        valid_features = engineer.calculate_all_features(valid_df)
        
        valid_output = output_dir / "valid_features.csv"
        valid_features.to_csv(valid_output, index=False)
        print(f"✅ 验证特征保存: {valid_output}")
        print(f"   形状: {valid_features.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 特征工程完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
