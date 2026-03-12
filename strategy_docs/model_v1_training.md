# 模型 V1 训练文档

> 训练日期：2026-03-12
> 训练者：牧心
> 模型类型：LightGBM/XGBoost 多因子选股模型

---

## 1. 环境配置

### 安装依赖
```bash
pip install pandas numpy lightgbm==4.6.0 xgboost==3.2.0 scikit-learn pyyaml tushare
```

### 项目结构
```
mkdir -p config tushare_provider workflow results data/cache
touch tushare_provider/__init__.py workflow/__init__.py
```

---

## 2. 配置文件

### config/workflow_config.yaml
```yaml
data:
  tushare_token: "你的Token"
  cache_path: "data/cache"
  train_start: "2020-01-01"
  train_end: "2024-08-31"
  valid_start: "2024-09-01"
  valid_end: "2025-08-31"

output:
  model_dir: "results"
```

---

## 3. 数据获取模块

### tushare_provider/tushare_fetcher.py

**功能：**
- 获取股票列表（沪深300/中证500/中证1000）
- 获取日线价格数据
- 本地缓存机制

**关键类：**
- `TushareDataFetcher`：数据获取器
  - `get_stock_list(market)`：获取指数成分股
  - `get_daily_prices(symbol, sd, ed)`：获取单只股票日线
  - `get_daily_prices_batch(symbols, sd, ed)`：批量获取

**缓存策略：**
- 股票列表缓存：`stock_list_{market}.pkl`
- 价格数据缓存：`daily_{symbol}_{sd}_{ed}.pkl`

---

## 4. 训练参数

### 数据划分
- **训练集**：2020-01-01 至 2024-08-31
- **验证集**：2024-09-01 至 2025-08-31

### 股票池
- 中证500成分股（默认）
- 可选：沪深300、中证1000

---

## 5. 后续步骤

- [ ] 特征工程
- [ ] 模型训练
- [ ] 回测验证
- [ ] 实盘部署

---

_Updated: 2026-03-12_
