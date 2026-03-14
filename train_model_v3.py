#!/usr/bin/env python3
"""
V3模型训练 - 预测超额收益（Alpha）
- 计算相对CSI500的超额收益作为Target
- 让模型专注于寻找Alpha
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent))

# 复用现有的模型训练器
from workflow.model_ensemble import V7Trainer


def calculate_excess_return(df):
    """
    计算每只股票相对于CSI500的超额收益
    """
    # 按日期分组，计算每日CSI500指数收益（成分股平均）
    index_returns = df.groupby('date')['pct_chg'].mean().reset_index()
    index_returns.columns = ['date', 'index_return']
    
    # 合并到原始数据
    df = df.merge(index_returns, on='date', how='left')
    
    # 计算超额收益 = 股票收益 - 指数收益
    df['excess_return'] = df['pct_chg'] - df['index_return']
    
    return df


def get_safe_features(df):
    """获取安全的特征列表"""
    exclude_cols = [
        'ts_code', 'date', 'open', 'high', 'low', 'close',
        'pre_close', 'change', 'pct_chg', 'volume', 'amount', 'symbol',
        'label', 'index_return', 'excess_return', 'stock_name'
    ]
    
    risky_patterns = ['return_1d', 'next_', 'future_', 'target_']
    
    safe_features = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(pattern in col for pattern in risky_patterns):
            print(f"  ⚠️ 排除风险特征: {col}")
            continue
        safe_features.append(col)
    
    return safe_features


def prepare_data_with_excess(train_file, valid_file):
    """准备训练数据（使用超额收益作为标签）"""
    print("\n" + "="*60)
    print("📥 加载数据（V3 - 超额收益版本）")
    print("="*60)
    
    # 读取数据
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    print(f"原始训练集: {len(train_df)} 条")
    print(f"原始验证集: {len(valid_df)} 条")
    
    # 计算超额收益
    print("\n🔧 计算超额收益...")
    train_df = calculate_excess_return(train_df)
    valid_df = calculate_excess_return(valid_df)
    
    # 创建标签（次日超额收益）
    print("🏷️ 创建超额收益标签...")
    train_df['label'] = train_df.groupby('symbol')['excess_return'].shift(-1)
    valid_df['label'] = valid_df.groupby('symbol')['excess_return'].shift(-1)
    
    # 获取安全特征
    feature_cols = get_safe_features(train_df)
    print(f"\n📊 安全特征数: {len(feature_cols)}")
    
    # 删除缺失值
    train_df = train_df.dropna(subset=feature_cols + ['label'])
    valid_df = valid_df.dropna(subset=feature_cols + ['label'])
    
    print(f"清洗后 - 训练集: {len(train_df)}, 验证集: {len(valid_df)}")
    
    # 统计超额收益
    print(f"\n📈 超额收益统计:")
    print(f"  训练集均值: {train_df['label'].mean()*100:.4f}%")
    print(f"  训练集标准差: {train_df['label'].std()*100:.4f}%")
    print(f"  验证集均值: {valid_df['label'].mean()*100:.4f}%")
    print(f"  验证集标准差: {valid_df['label'].std()*100:.4f}%")
    
    # 准备数据
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['label']
    
    return X_train, y_train, X_valid, y_valid, feature_cols


def train_model_v3():
    """训练V3模型（预测超额收益）"""
    print("="*60)
    print("🚀 V3模型训练 - Alpha预测版")
    print("="*60)
    
    # 数据路径
    data_dir = Path("data/processed")
    train_file = data_dir / "train_features.csv"
    valid_file = data_dir / "valid_features.csv"
    
    if not train_file.exists():
        print(f"❌ 找不到训练数据: {train_file}")
        return
    
    # 准备数据（超额收益标签）
    X_train, y_train, X_valid, y_valid, feature_cols = prepare_data_with_excess(train_file, valid_file)
    
    # 训练模型
    print("\n" + "="*60)
    print("🤖 训练Alpha预测模型")
    print("="*60)
    
    trainer = V7Trainer("csi500_v3_alpha")
    
    try:
        ic = trainer.train(X_train, y_train, X_valid, y_valid)
        
        # 保存模型（V3版本）
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model_csi500_v3.pkl"
        
        model_data = {
            'models': trainer.models,
            'weights': trainer.weights,
            'features': feature_cols,
            'ensemble_ic': ic,
            'version': 'v3_alpha',
            'trainer_class': 'V7Trainer',
            'feature_count': len(feature_cols),
            'train_samples': len(X_train),
            'valid_samples': len(X_valid),
            'target': 'excess_return',
            'description': '预测相对CSI500的超额收益'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("\n" + "="*60)
        print("✅ V3模型训练完成!")
        print("="*60)
        print(f"模型版本: V3 (Alpha预测)")
        print(f"预测目标: 超额收益 (股票收益 - CSI500收益)")
        print(f"集成IC: {ic:.4f}")
        print(f"特征数: {len(feature_cols)}")
        print(f"模型保存: {model_path}")
        print(f"\n📝 注意: V1/V2模型未被覆盖")
        
        # 保存特征列表
        feature_file = model_dir / "features_v3.txt"
        with open(feature_file, 'w') as f:
            f.write(f"# CSI500 V3 Model Features (Alpha Prediction)\n")
            f.write(f"# IC: {ic:.4f}\n")
            f.write(f"# Target: Excess Return (Stock Return - CSI500 Return)\n")
            f.write(f"# Feature Count: {len(feature_cols)}\n")
            f.write("\n")
            for feat in feature_cols:
                f.write(f"{feat}\n")
        print(f"特征列表: {feature_file}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    train_model_v3()
