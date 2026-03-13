#!/usr/bin/env python3
"""
简化版模型训练 - 只使用 LightGBM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))


def calculate_ic(y_true, y_pred):
    """计算信息系数 IC"""
    return np.corrcoef(y_true, y_pred)[0, 1]


def train_lgb_model(X_train, y_train, X_valid, y_valid):
    """训练 LightGBM 模型"""
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # 参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 训练
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)]
    )
    
    # 预测
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    
    # 计算指标
    ic = calculate_ic(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    return model, ic, rmse


def main():
    """主函数"""
    print("="*60)
    print("🚀 LightGBM 模型训练")
    print("="*60)
    
    # 读取数据
    data_dir = Path("data/processed")
    train_file = data_dir / "train_features.csv"
    valid_file = data_dir / "valid_features.csv"
    
    print(f"\n📥 读取数据...")
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    print(f"✅ 训练数据: {len(train_df)} 条")
    print(f"✅ 验证数据: {len(valid_df)} 条")
    
    # 获取特征列
    exclude_cols = ['ts_code', 'date', 'open', 'high', 'low', 'close', 
                    'pre_close', 'change', 'pct_chg', 'volume', 'amount', 'symbol']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"\n📊 特征数量: {len(feature_cols)}")
    
    # 创建标签
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
    model, ic, rmse = train_lgb_model(X_train, y_train, X_valid, y_valid)
    
    # 保存模型
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model_csi500_lgb.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'ic': ic,
            'rmse': rmse
        }, f)
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 重要特征:")
    print(importance.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"✅ 模型训练完成!")
    print(f"📊 IC: {ic:.4f}")
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📁 模型保存: {model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
