#!/usr/bin/env python3
"""
V4模型训练 - Alpha预测 + 市值中性化 + 排序学习

改进点：
1. Alpha Label: 预测相对CSI500的超额收益
2. 市值中性化: 截面回归去除市值影响
3. 排序学习: LambdaRank优化排序
4. 评估指标: Rank IC + Top5%超额收益
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import sys

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')


class FeatureNeutralizer:
    """特征市值中性化器"""
    
    def __init__(self):
        self.neutralized_features = []
    
    def neutralize_features(self, df, features_to_neutralize, market_cap_col='market_cap_proxy'):
        """
        对特征进行市值中性化
        
        Args:
            df: 数据框
            features_to_neutralize: 需要中性化的特征列表
            market_cap_col: 市值代理列名
            
        Returns:
            DataFrame: 中性化后的数据
        """
        print(f"🔧 开始市值中性化...")
        df = df.copy()
        
        # 计算Log市值
        df['log_market_cap'] = np.log(df[market_cap_col] + 1e-8)
        
        # 按日期分组进行截面回归
        neutralized_count = 0
        for date, group in df.groupby('date'):
            if len(group) < 10:  # 样本太少跳过
                continue
            
            X = group[['log_market_cap']].values
            
            for feat in features_to_neutralize:
                if feat not in group.columns:
                    continue
                
                y = group[feat].values
                
                # 去除NaN
                mask = ~(np.isnan(y) | np.isnan(X[:, 0]))
                if mask.sum() < 10:
                    continue
                
                X_valid = X[mask]
                y_valid = y[mask]
                
                # 截面回归
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                
                # 预测并计算残差
                y_pred = model.predict(X_valid)
                residual = y_valid - y_pred
                
                # 保存残差
                df.loc[group.index[mask], f'{feat}_neutral'] = residual
                neutralized_count += 1
        
        print(f"✅ 完成 {neutralized_count} 个特征的中性化")
        return df


class AlphaLabelCalculator:
    """Alpha标签计算器"""
    
    @staticmethod
    def calculate_alpha_label(df, horizon=5):
        """
        计算Alpha标签（股票收益 - CSI500收益）
        
        Args:
            df: 数据框
            horizon: 预测周期（天数）
            
        Returns:
            DataFrame: 添加Alpha标签后的数据
        """
        print(f"🔧 计算Alpha标签（{horizon}日）...")
        df = df.copy()
        
        # 计算指数收益（成分股平均）
        index_returns = df.groupby('date')['close'].mean().reset_index()
        index_returns.columns = ['date', 'index_close']
        index_returns[f'index_return_{horizon}d'] = index_returns['index_close'].shift(-horizon) / index_returns['index_close'] - 1
        
        # 合并到原始数据
        df = df.merge(index_returns[['date', f'index_return_{horizon}d']], on='date', how='left')
        
        # 按股票计算未来收益和Alpha
        def calc_alpha(group):
            group = group.sort_values('date')
            # 股票未来N日收益
            group[f'stock_return_{horizon}d'] = group['close'].shift(-horizon) / group['close'] - 1
            # Alpha = 股票收益 - 指数收益
            group[f'alpha_{horizon}d'] = group[f'stock_return_{horizon}d'] - group[f'index_return_{horizon}d']
            return group
        
        df = df.groupby('symbol', group_keys=False).apply(calc_alpha)
        
        # 统计
        alpha_mean = df[f'alpha_{horizon}d'].mean()
        alpha_std = df[f'alpha_{horizon}d'].std()
        print(f"✅ Alpha标签统计: 均值={alpha_mean*100:.4f}%, 标准差={alpha_std*100:.4f}%")
        
        return df


class LambdaRankTrainer:
    """LambdaRank排序学习训练器"""
    
    def __init__(self, pool="csi500"):
        self.pool = pool
        self.models = {}
        self.weights = {}
        self.metrics = {}
    
    def _calculate_ic(self, y_true, y_pred):
        """计算IC（Pearson相关系数）"""
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        if m.sum() < 10:
            return 0
        return np.corrcoef(y_true[m], y_pred[m])[0, 1]
    
    def _calculate_rank_ic(self, y_true, y_pred):
        """计算Rank IC（Spearman秩相关系数）"""
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        if m.sum() < 10:
            return 0
        corr, _ = spearmanr(y_true[m], y_pred[m])
        return corr
    
    def _calculate_top5_excess(self, y_true, y_pred):
        """计算预测得分前5%的平均超额收益"""
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        if m.sum() < 20:
            return 0
        
        y_true_clean = y_true[m]
        y_pred_clean = y_pred[m]
        
        # 选前5%
        threshold = np.percentile(y_pred_clean, 95)
        top5_mask = y_pred_clean >= threshold
        
        if top5_mask.sum() == 0:
            return 0
        
        return y_true_clean[top5_mask].mean()
    
    def _prepare_ranking_labels(self, y, group_sizes):
        """
        将连续标签转换为排名标签（每组内按分位数分桶）
        
        Args:
            y: 连续标签（Alpha）
            group_sizes: 每组的大小
            
        Returns:
            array: 整数排名标签（0-4，5个等级）
        """
        y_rank = np.zeros(len(y), dtype=int)
        idx = 0
        
        for size in group_sizes:
            group_y = y[idx:idx+size]
            
            # 按分位数分桶（5个等级）
            # 使用qcut将数据分为5个等级
            try:
                ranks = pd.qcut(group_y, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
                y_rank[idx:idx+size] = ranks.astype(int)
            except:
                # 如果分桶失败（数据太集中），使用简单的排名
                ranks = pd.cut(group_y, bins=5, labels=[0, 1, 2, 3, 4])
                y_rank[idx:idx+size] = ranks.astype(int)
            
            idx += size
        
        return y_rank
    
    def train(self, X_train, y_train, X_valid, y_valid, date_train, date_valid):
        """
        训练LambdaRank模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签（Alpha）
            X_valid: 验证特征
            y_valid: 验证标签（Alpha）
            date_train: 训练日期（用于分组）
            date_valid: 验证日期（用于分组）
        """
        from lightgbm import LGBMRanker, early_stopping, log_evaluation
        
        print("\n" + "="*60)
        print("🚀 训练LambdaRank排序模型")
        print("="*60)
        
        # 准备分组数据
        group_train = pd.Series(date_train).groupby(date_train).size().values
        group_valid = pd.Series(date_valid).groupby(date_valid).size().values
        
        print(f"训练组数: {len(group_train)}, 验证组数: {len(group_valid)}")
        
        # 将连续标签转换为排名标签
        print("🔧 转换标签为排名等级...")
        y_train_rank = self._prepare_ranking_labels(y_train.values, group_train)
        y_valid_rank = self._prepare_ranking_labels(y_valid.values, group_valid)
        
        print(f"标签分布: 训练集{y_train_rank.min()}-{y_train_rank.max()}, 验证集{y_valid_rank.min()}-{y_valid_rank.max()}")
        
        # LambdaRank模型
        print("\n🎯 Training LambdaRank...")
        ranker = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            ndcg_at=[5, 10, 20],
            learning_rate=0.02,
            n_estimators=500,
            max_depth=6,
            num_leaves=64,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=0.5,
            min_child_samples=30,
            label_gain=[0, 1, 3, 7, 15],  # 5个等级的增益（指数增长）
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        ranker.fit(
            X_train, y_train_rank,
            group=group_train,
            eval_set=[(X_valid, y_valid_rank)],
            eval_group=[group_valid],
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=10)
            ]
        )
        
        self.models['lambdarank'] = ranker
        
        # 预测并评估
        y_pred = ranker.predict(X_valid)
        
        # 计算评估指标
        ic = self._calculate_ic(y_valid.values, y_pred)
        rank_ic = self._calculate_rank_ic(y_valid.values, y_pred)
        top5_excess = self._calculate_top5_excess(y_valid.values, y_pred)
        
        self.metrics = {
            'ic': ic,
            'rank_ic': rank_ic,
            'top5_excess': top5_excess
        }
        
        print(f"\n📊 验证集指标:")
        print(f"  IC: {ic:.4f}")
        print(f"  Rank IC: {rank_ic:.4f}")
        print(f"  Top5%超额收益: {top5_excess*100:.4f}%")
        
        return self.metrics
    
    def save(self, path, features, metrics):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'features': features,
                'metrics': metrics,
                'type': f'v4_alpha_lambdarank',
                'version': '4.0'
            }, f)
        
        print(f"✅ Model saved: {path}")


def prepare_data_v4():
    """准备V4训练数据"""
    print("="*60)
    print("📥 准备V4训练数据")
    print("="*60)
    
    # 加载数据
    data_dir = Path("data/processed")
    train_file = data_dir / "train_features.csv"
    valid_file = data_dir / "valid_features.csv"
    
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    print(f"原始数据: 训练集{len(train_df)}, 验证集{len(valid_df)}")
    
    # 合并数据用于计算指数收益
    combined = pd.concat([train_df, valid_df], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    
    # 计算Alpha标签
    alpha_calc = AlphaLabelCalculator()
    combined = alpha_calc.calculate_alpha_label(combined, horizon=5)
    
    # 分离训练集和验证集
    train_max_date = pd.to_datetime(train_df['date']).max()
    train_mask = combined['date'] <= train_max_date
    train_df = combined[train_mask].copy()
    valid_df = combined[~train_mask].copy()
    
    print(f"Alpha标签数据: 训练集{len(train_df)}, 验证集{len(valid_df)}")
    
    # 特征中性化
    # 选择需要中性化的特征（技术面特征）
    features_to_neutralize = [
        'return_5d', 'return_10d', 'return_20d', 'return_60d',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'rsi_6', 'rsi_12', 'rsi_24',
        'macd', 'macd_signal'
    ]
    
    # 只保留存在的特征
    features_to_neutralize = [f for f in features_to_neutralize if f in train_df.columns]
    
    neutralizer = FeatureNeutralizer()
    train_df = neutralizer.neutralize_features(train_df, features_to_neutralize)
    valid_df = neutralizer.neutralize_features(valid_df, features_to_neutralize)
    
    # 准备特征列表
    # 使用中性化后的特征 + 未中性化的特征
    neutral_features = [f'{f}_neutral' for f in features_to_neutralize if f'{f}_neutral' in train_df.columns]
    other_features = ['volume_ratio', 'turnover_ratio', 'bb_position', 'mfi']
    other_features = [f for f in other_features if f in train_df.columns]
    
    feature_cols = neutral_features + other_features
    
    print(f"特征列表: {len(feature_cols)}个")
    print(f"  中性化特征: {len(neutral_features)}个")
    print(f"  其他特征: {len(other_features)}个")
    
    # 删除缺失值
    label_col = 'alpha_5d'
    train_clean = train_df.dropna(subset=[label_col] + feature_cols)
    valid_clean = valid_df.dropna(subset=[label_col] + feature_cols)
    
    print(f"清洗后: 训练集{len(train_clean)}, 验证集{len(valid_clean)}")
    
    # 准备数据
    X_train = train_clean[feature_cols]
    y_train = train_clean[label_col]
    X_valid = valid_clean[feature_cols]
    y_valid = valid_clean[label_col]
    date_train = train_clean['date']
    date_valid = valid_clean['date']
    
    return (X_train, y_train, X_valid, y_valid, 
            date_train, date_valid, feature_cols)


def train_model_v4():
    """训练V4模型"""
    print("="*60)
    print("🚀 V4模型训练 - Alpha + 中性化 + LambdaRank")
    print("="*60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 准备数据
    (X_train, y_train, X_valid, y_valid,
     date_train, date_valid, feature_cols) = prepare_data_v4()
    
    # 训练LambdaRank
    trainer = LambdaRankTrainer("csi500_v4")
    metrics = trainer.train(X_train, y_train, X_valid, y_valid, date_train, date_valid)
    
    # 保存模型
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model_csi500_v4.pkl"
    
    trainer.save(model_path, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("✅ V4模型训练完成!")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"\n最终指标:")
    print(f"  IC: {metrics['ic']:.4f}")
    print(f"  Rank IC: {metrics['rank_ic']:.4f}")
    print(f"  Top5%超额收益: {metrics['top5_excess']*100:.4f}%")
    
    return model_path


if __name__ == '__main__':
    train_model_v4()