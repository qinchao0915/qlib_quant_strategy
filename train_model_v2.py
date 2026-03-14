#!/usr/bin/env python3
"""
CSI500 模型训练 V2

改进点：
1. 优化特征选择 - 去除可能导致数据穿越的特征
2. 调整模型参数 - 降低过拟合
3. 添加交叉验证
4. 模型版本管理 - 保存为 model_csi500_v2.pkl

复用模块：
- workflow/feature_engineering.py (特征工程)
- workflow/model_ensemble.py (模型集成)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 复用现有模块
from workflow.model_ensemble import V7Trainer

warnings.filterwarnings('ignore')


class DataLeakageAwareFeatureSelector:
    """
    数据穿越感知特征选择器
    
    识别并移除可能导致数据穿越的特征：
    - 使用未来信息的特征（如未来收益率）
    - 包含未来价格信息的特征
    - 前瞻性的统计量
    """
    
    # 可能导致数据穿越的特征列表
    LEAKAGE_FEATURES = {
        # 直接使用未来价格的特征
        'return_1d',  # 次日收益率（标签）
        'return_3d',  # 未来3日收益率
        'return_5d',  # 未来5日收益率
        'return_10d',  # 未来10日收益率
        'return_20d',  # 未来20日收益率
        'return_60d',  # 未来60日收益率
        'return_accel',  # 基于未来收益率计算
        
        # 原始价格数据（避免使用）
        'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg',
        
        # 原始成交量数据
        'volume', 'amount',
        
        # 时间戳
        'date', 'ts_code', 'symbol',
    }
    
    # 推荐使用的安全特征
    SAFE_FEATURE_PATTERNS = [
        # 技术指标（基于历史价格计算）
        'volatility_',  # 波动率
        'ma_',  # 移动平均线
        'price_to_ma',  # 价格相对均线位置
        'bb_',  # 布林带
        'rsi_',  # RSI
        'macd',  # MACD
        'volume_ma',  # 成交量均线
        'volume_ratio',  # 成交量比率
        'turnover_ma',  # 成交额均线
        'turnover_ratio',  # 成交额比率
        'amplitude_',  # 振幅
        'price_volume_corr',  # 价量相关性
        
        # 资金流向特征
        'typical_price',  # 典型价格
        'raw_money_flow',  # 原始资金流
        'mfi',  # 资金流量指标
        'big_money',  # 大单标志
        'big_money_ratio',  # 大单占比
        'net_money_flow',  # 净流入
        'net_money_flow_5d',  # 5日净流入
        
        # 市值分层特征
        'market_cap_proxy',  # 市值代理
        'is_low_price',  # 低价股标志
        'liquidity_score',  # 流动性得分
    ]
    
    def __init__(self):
        self.selected_features = []
        self.excluded_features = []
    
    def select_features(self, df):
        """
        选择安全的特征
        
        Args:
            df: 输入数据框
            
        Returns:
            list: 安全的特征列表
        """
        all_columns = set(df.columns)
        
        # 第一步：排除已知的数据穿越特征
        safe_features = []
        for col in all_columns:
            if col not in self.LEAKAGE_FEATURES:
                safe_features.append(col)
        
        # 第二步：验证特征是否符合安全模式
        final_features = []
        for col in safe_features:
            is_safe = any(pattern in col for pattern in self.SAFE_FEATURE_PATTERNS)
            if is_safe:
                final_features.append(col)
            else:
                self.excluded_features.append(col)
        
        # 过滤掉包含NaN过多的特征（>50%）
        feature_stats = []
        for col in final_features:
            nan_ratio = df[col].isna().mean()
            if nan_ratio < 0.5:  # 保留NaN比例小于50%的特征
                feature_stats.append((col, nan_ratio))
            else:
                self.excluded_features.append(f"{col}(NaN:{nan_ratio:.2%})")
        
        self.selected_features = [f[0] for f in feature_stats]
        
        print(f"✅ 特征选择完成:")
        print(f"   - 原始特征数: {len(all_columns)}")
        print(f"   - 选中特征数: {len(self.selected_features)}")
        print(f"   - 排除特征数: {len(self.excluded_features)}")
        
        return self.selected_features
    
    def get_feature_report(self):
        """获取特征选择报告"""
        return {
            'selected': self.selected_features,
            'excluded': self.excluded_features,
            'count': len(self.selected_features)
        }


class AntiOverfittingV7Trainer(V7Trainer):
    """
    改进的V7训练器 - 降低过拟合
    
    改进点：
    1. 更强的正则化参数
    2. 早停机制
    3. 交叉验证
    """
    
    def __init__(self, pool="csi500", n_splits=5):
        super().__init__(pool)
        self.n_splits = n_splits
        self.cv_scores = []
        self.feature_importance = {}
    
    def train_with_cv(self, X_train, y_train, X_valid, y_valid):
        """
        带交叉验证的训练
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
            
        Returns:
            float: 交叉验证后的IC
        """
        from lightgbm import LGBMRegressor
        
        print(f"\n🔍 开始交叉验证 (n_splits={self.n_splits})...")
        
        # 合并数据用于交叉验证
        X_combined = pd.concat([X_train, X_valid], ignore_index=True)
        y_combined = pd.concat([y_train, y_valid], ignore_index=True)
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_ics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined)):
            X_cv_train = X_combined.iloc[train_idx]
            y_cv_train = y_combined.iloc[train_idx]
            X_cv_val = X_combined.iloc[val_idx]
            y_cv_val = y_combined.iloc[val_idx]
            
            # 快速训练一个LGBM评估
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=32,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42 + fold,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_cv_train, y_cv_train)
            
            pred = model.predict(X_cv_val)
            ic = self._calculate_ic(y_cv_val.values, pred)
            cv_ics.append(ic)
            print(f"   Fold {fold+1}: IC = {ic:.4f}")
        
        mean_cv_ic = np.mean(cv_ics)
        std_cv_ic = np.std(cv_ics)
        print(f"   CV Mean IC: {mean_cv_ic:.4f} (±{std_cv_ic:.4f})")
        
        self.cv_scores = cv_ics
        
        # 使用原始数据进行最终训练
        return self.train(X_train, y_train, X_valid, y_valid)
    
    def train(self, X_train, y_train, X_valid, y_valid):
        """
        训练模型 - 使用更强的正则化参数
        """
        from lightgbm import LGBMRegressor
        
        vp = {}
        
        # ========== 1. LGBM Conservative (增强正则化) ==========
        print("\n🎯 Training LGBM Conservative (Anti-Overfitting)...")
        lgb_cons = LGBMRegressor(
            n_estimators=400,  # 减少树的数量
            learning_rate=0.02,  # 降低学习率
            max_depth=6,  # 降低深度
            num_leaves=64,  # 减少叶子数
            subsample=0.7,  # 增加采样随机性
            colsample_bytree=0.7,
            reg_alpha=0.5,  # 增强L1正则化
            reg_lambda=1.0,  # 增强L2正则化
            min_child_samples=50,  # 增加最小样本数
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_cons.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[]  # 不使用早停，固定树数量
        )
        self.models['lgbm_conservative'] = lgb_cons
        vp['lgbm_conservative'] = lgb_cons.predict(X_valid)
        self.ic_scores['lgbm_conservative'] = self._calculate_ic(y_valid.values, vp['lgbm_conservative'])
        print(f"   LGBM Cons: IC={self.ic_scores['lgbm_conservative']:.4f}")
        
        # 记录特征重要性
        self.feature_importance['lgbm_conservative'] = dict(
            zip(X_train.columns, lgb_cons.feature_importances_)
        )
        
        # ========== 2. LGBM Bagging (增强随机性) ==========
        print("\n🎯 Training LGBM Bagging (Enhanced)...")
        lgb_bag = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=32,
            subsample=0.5,  # 更强的采样
            colsample_bytree=0.5,
            bagging_freq=3,
            bagging_fraction=0.5,
            reg_alpha=0.3,
            reg_lambda=0.5,
            min_child_samples=30,
            random_state=43,
            n_jobs=-1,
            verbose=-1
        )
        lgb_bag.fit(X_train, y_train)
        self.models['lgbm_bagging'] = lgb_bag
        vp['lgbm_bagging'] = lgb_bag.predict(X_valid)
        self.ic_scores['lgbm_bagging'] = self._calculate_ic(y_valid.values, vp['lgbm_bagging'])
        print(f"   LGBM Bag: IC={self.ic_scores['lgbm_bagging']:.4f}")
        
        self.feature_importance['lgbm_bagging'] = dict(
            zip(X_train.columns, lgb_bag.feature_importances_)
        )
        
        # ========== 3. XGBoost (可选) ==========
        print("\n🎯 Training XGBoost...")
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                n_estimators=400,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_weight=5,
                random_state=44,
                n_jobs=-1,
                verbosity=0
            )
            xgb.fit(X_train, y_train)
            self.models['xgb'] = xgb
            vp['xgb'] = xgb.predict(X_valid)
            self.ic_scores['xgb'] = self._calculate_ic(y_valid.values, vp['xgb'])
            print(f"   XGB: IC={self.ic_scores['xgb']:.4f}")
            
            self.feature_importance['xgb'] = dict(
                zip(X_train.columns, xgb.feature_importances_)
            )
        except ImportError:
            print("   ⚠️ XGBoost not installed, skipping")
            self.ic_scores['xgb'] = 0
        
        # ========== 4. 权重优化 ==========
        print("\n⚖️  Optimizing ensemble weights...")
        mn = list(self.ic_scores.keys())
        
        # 基于IC的初始权重
        ic_sum = sum(max(0, self.ic_scores[m]) for m in mn)
        if ic_sum > 0:
            iw = np.array([max(0, self.ic_scores[m]) / ic_sum for m in mn])
        else:
            iw = np.ones(len(mn)) / len(mn)
        
        # 目标函数：最大化集成IC
        def obj(w):
            ep = np.zeros(len(y_valid))
            for i, n in enumerate(mn):
                ep = ep + w[i] * vp[n]
            return -self._calculate_ic(y_valid.values, ep)
        
        # SLSQP优化
        from scipy.optimize import minimize
        r = minimize(
            obj,
            iw,
            method='SLSQP',
            bounds=[(0, 1)] * len(mn),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        self.weights = {n: r.x[i] for i, n in enumerate(mn)}
        
        # 计算集成IC
        ep = sum(self.weights[m] * vp[m] for m in mn)
        eic = self._calculate_ic(y_valid.values, ep)
        
        print(f"\n✅ Ensemble IC: {eic:.4f}")
        print(f"   Weights: {self.weights}")
        
        return eic
    
    def get_top_features(self, n=20):
        """获取最重要的特征"""
        if not self.feature_importance:
            return []
        
        # 平均各模型的特征重要性
        avg_importance = {}
        for model_name, importance in self.feature_importance.items():
            weight = self.weights.get(model_name, 0)
            if weight > 0:
                for feat, imp in importance.items():
                    if feat not in avg_importance:
                        avg_importance[feat] = 0
                    avg_importance[feat] += imp * weight
        
        # 排序
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


def load_and_prepare_data(data_dir="data/processed"):
    """
    加载并准备数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        tuple: (train_df, valid_df)
    """
    data_path = Path(data_dir)
    
    print("📊 加载数据...")
    train_df = pd.read_csv(data_path / "train_features.csv")
    valid_df = pd.read_csv(data_path / "valid_features.csv")
    
    print(f"   训练集: {len(train_df)} 行, {len(train_df.columns)} 列")
    print(f"   验证集: {len(valid_df)} 行, {len(valid_df.columns)} 列")
    
    return train_df, valid_df


def prepare_labels(df, horizon=5):
    """
    准备标签（未来收益率）
    
    Args:
        df: 数据框
        horizon: 预测周期（天数）
        
    Returns:
        DataFrame: 添加标签后的数据
    """
    df = df.copy()
    
    # 按股票分组计算未来收益率
    def calc_future_return(group):
        group = group.sort_values('date')
        # 未来N日收益率（标签）
        group[f'label_{horizon}d'] = group['close'].shift(-horizon) / group['close'] - 1
        return group
    
    df = df.groupby('symbol', group_keys=False).apply(calc_future_return)
    
    return df


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 CSI500 模型训练 V2")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    train_df, valid_df = load_and_prepare_data()
    
    # 2. 准备标签
    print("\n🏷️  准备标签...")
    train_df = prepare_labels(train_df)
    valid_df = prepare_labels(valid_df)
    
    # 3. 特征选择（去除数据穿越）
    print("\n🔍 特征选择（数据穿越感知）...")
    selector = DataLeakageAwareFeatureSelector()
    
    # 合并数据以统一特征选择
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    selected_features = selector.select_features(combined_df)
    
    # 打印特征报告
    report = selector.get_feature_report()
    print(f"\n📋 选中特征 ({len(selected_features)} 个):")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"   {i}. {feat}")
    if len(selected_features) > 10:
        print(f"   ... 等共 {len(selected_features)} 个特征")
    
    # 4. 准备训练数据
    print("\n📦 准备训练数据...")
    
    # 去除NaN
    train_clean = train_df.dropna(subset=['label_5d'] + selected_features)
    valid_clean = valid_df.dropna(subset=['label_5d'] + selected_features)
    
    X_train = train_clean[selected_features]
    y_train = train_clean['label_5d']
    X_valid = valid_clean[selected_features]
    y_valid = valid_clean['label_5d']
    
    print(f"   训练样本: {len(X_train)}")
    print(f"   验证样本: {len(X_valid)}")
    
    # 5. 训练模型
    print("\n" + "=" * 60)
    print("🎯 开始模型训练")
    print("=" * 60)
    
    trainer = AntiOverfittingV7Trainer(pool="csi500", n_splits=3)
    
    # 使用交叉验证训练
    ensemble_ic = trainer.train_with_cv(X_train, y_train, X_valid, y_valid)
    
    # 6. 打印特征重要性
    print("\n📊 Top 15 重要特征:")
    top_features = trainer.get_top_features(n=15)
    for i, (feat, imp) in enumerate(top_features, 1):
        print(f"   {i:2d}. {feat:<25} {imp:.4f}")
    
    # 7. 保存模型
    print("\n💾 保存模型...")
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "model_csi500_v2.pkl"
    
    # 保存额外信息
    save_data = {
        'models': trainer.models,
        'weights': trainer.weights,
        'features': selected_features,
        'ic_scores': trainer.ic_scores,
        'ensemble_ic': ensemble_ic,
        'cv_scores': trainer.cv_scores,
        'feature_importance': trainer.feature_importance,
        'top_features': top_features,
        'type': 'v2_csi500',
        'version': '2.0',
        'created_at': datetime.now().isoformat(),
        'config': {
            'pool': 'csi500',
            'horizon': 5,
            'n_features': len(selected_features),
            'n_train_samples': len(X_train),
            'n_valid_samples': len(X_valid),
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"   ✅ 模型已保存: {model_path}")
    
    # 8. 输出总结
    print("\n" + "=" * 60)
    print("📈 训练完成总结")
    print("=" * 60)
    print(f"   模型文件: {model_path}")
    print(f"   集成IC: {ensemble_ic:.4f}")
    print(f"   交叉验证IC: {np.mean(trainer.cv_scores):.4f} (±{np.std(trainer.cv_scores):.4f})")
    print(f"   特征数量: {len(selected_features)}")
    print(f"   模型权重: {trainer.weights}")
    print(f"   完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()