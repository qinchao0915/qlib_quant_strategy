# 模型 V7 集成训练文档

> 训练日期：2026-03-12
> 训练者：牧心
> 模型类型：LightGBM + XGBoost 集成模型

---

## 1. 模型架构

### 集成策略
- **LGBM Conservative**: 保守型 LightGBM
- **LGBM Bagging**: Bagging LightGBM
- **XGBoost**: XGBoost 模型
- **权重优化**: 基于 IC 的 SLSQP 优化

### 模型参数

#### LGBM Conservative
```python
n_estimators=600
learning_rate=0.03
max_depth=10
num_leaves=200
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=0.1
```

#### LGBM Bagging
```python
n_estimators=400
learning_rate=0.04
max_depth=6
num_leaves=64
subsample=0.6
colsample_bytree=0.6
bagging_freq=5
bagging_fraction=0.6
```

#### XGBoost
```python
n_estimators=600
learning_rate=0.03
max_depth=10
subsample=0.8
colsample_bytree=0.8
```

---

## 2. 训练结果

### IC 表现
| 模型 | IC |
|------|-----|
| LGBM Conservative | 0.0582 |
| LGBM Bagging | 0.0421 |
| XGBoost | **0.0650** |
| **Ensemble** | **0.0650** |

### 最优权重
```python
{'xgb': 1.0, 'lgbm_conservative': 0.0, 'lgbm_bagging': 0.0}
```

**结论**：XGBoost 表现最好，集成后权重全部给 XGBoost

---

## 3. 特征工程

### 特征数量：53个

| 类别 | 数量 | 特征 |
|------|------|------|
| 动量 | 8 | return_1d, return_3d, return_5d, return_10d, return_20d, return_60d, return_accel, return_volatility |
| 波动率 | 6 | volatility_5d, volatility_10d, volatility_20d, volatility_60d, volatility_ratio, amplitude_20d |
| 均线 | 8 | ma5, ma10, ma20, ma60, price_to_ma5, price_to_ma10, price_to_ma20, price_to_ma60 |
| 布林带 | 3 | bb_position, bb_width, bb_upper_ratio |
| RSI | 4 | rsi_6, rsi_12, rsi_24, rsi_diff |
| MACD | 5 | macd, macd_signal, macd_hist, macd_cross, macd_trend |
| 成交量 | 8 | volume_ma5, volume_ma20, volume_ratio, turnover_ratio, volume_price_trend, volume_spike, money_flow, net_money_flow |
| 振幅 | 5 | amplitude_5d, amplitude_10d, amplitude_20d, amplitude_60d, amplitude_ratio |
| 资金流 | 5 | big_money_ratio, small_money_ratio, money_flow_ma5, money_flow_ma20, money_flow_trend |
| 市值 | 3 | market_cap, market_cap_rank, market_cap_log |

### Top 10 特征
1. `net_money_flow` - 净流入
2. `return_3d` - 3日收益
3. `volatility_20d` - 20日波动率
4. `turnover_ratio` - 换手率
5. `mfi` - 资金流量指标
6. `amplitude_20d` - 20日振幅
7. `return_accel` - 收益加速度
8. `price_to_ma20` - 价格/20日均线
9. `ma5_to_ma20` - 5日/20日均线
10. `big_money_ratio` - 大单比例

---

## 4. 训练配置

### 数据划分
- **训练集**：2020-01 至 2024-08
- **验证集**：2024-09 至 2025-08

### 股票池
- **CSI500**：597只（中证500成分股）

### 目标变量
- **未来5日收益**（label）
- **目标IC**：0.065

---

## 5. 关键文件

- `workflow/feature_engineering_v7.py` - 特征工程
- `workflow/model_ensemble_v7.py` - 模型训练
- `workflow/train_v7_csi500.py` - 主训练脚本
- `results/model_enhanced_v7_csi500.pkl` - 保存的模型

---

## 6. 使用方式

```bash
python workflow/train_v7_csi500.py
```

---

_Updated: 2026-03-12_
