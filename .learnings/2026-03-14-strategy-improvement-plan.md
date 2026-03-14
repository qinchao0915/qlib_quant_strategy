---
date: 2026-03-14
type: improvement
status: pending
severity: P1
---

## 改进点

当前模型交易过于频繁、持仓时间太短、缺乏大盘环境过滤

### 当前状况
- 每天交易99笔，换手率过高
- 100%持仓时间为1天（日内交易）
- 没有大盘过滤，熊市也满仓
- 未考虑交易成本（滑点+手续费）

### 改进目标
- 降低交易频率（目标：每周1-2次调仓）
- 延长持仓时间（目标：3-5天）
- 添加大盘环境过滤
- 考虑交易成本

---

## 具体改进方案

### 1. 降低交易频率

**现状**：每天选前20%股票（99只），次日全部换仓

**改进**：
```python
# 方案A：降低选股比例
n_select = max(1, int(len(group) * 0.05))  # 从20%降到5%（约25只）

# 方案B：设置持仓门槛
# 只有当预测得分 > 阈值时才买入
threshold = pred_score.mean() + 1.5 * pred_score.std()
top_stocks = group[group['pred'] > threshold]

# 方案C：定期调仓（每周）
# 只在每周一进行调仓，其他时间持有不动
if date.weekday() == 0:  # 周一
    rebalance = True
```

### 2. 延长持仓时间

**现状**：T+1日内交易（开盘买，收盘卖）

**改进**：
```python
# 方案A：持有3天
holding_days = 3
# 买入后持有3天再卖出

# 方案B：趋势跟踪
# 当预测得分持续为正时继续持有
# 当预测得分转负或达到止损位时卖出

# 方案C：动态持仓
# 根据市场波动率调整持仓时间
# 高波动：缩短持仓
# 低波动：延长持仓
```

### 3. 添加大盘环境过滤

**现状**：无论大盘涨跌都满仓交易

**改进**：
```python
# 大盘过滤器
def market_filter(index_data):
    """
    返回仓位建议：0-1之间
    """
    # 技术面过滤
    ma20 = index_data['close'].rolling(20).mean()
    ma60 = index_data['close'].rolling(60).mean()
    
    # 大盘在60日均线上方，且20日均线向上
    if index_data['close'] > ma20 and ma20 > ma60:
        return 1.0  # 满仓
    elif index_data['close'] > ma60:
        return 0.5  # 半仓
    else:
        return 0.0  # 空仓

# 在回测中应用
position_ratio = market_filter(csi500_index)
if position_ratio == 0:
    # 空仓，不交易
    continue
elif position_ratio == 0.5:
    # 半仓，减少选股数量
    n_select = n_select // 2
```

### 4. 考虑交易成本

**现状**：未考虑滑点和手续费

**改进**：
```python
# 交易成本模型
def calculate_cost(price, volume, is_buy=True):
    """
    计算交易成本
    """
    # 手续费：万分之3（买卖双向）
    commission_rate = 0.0003
    # 滑点：0.1%
    slippage = 0.001
    
    commission = price * volume * commission_rate
    slippage_cost = price * volume * slippage
    
    if is_buy:
        # 买入：价格上浮（滑点）
        effective_price = price * (1 + slippage)
    else:
        # 卖出：价格下浮（滑点）
        effective_price = price * (1 - slippage)
    
    return effective_price, commission + slippage_cost

# 在回测中应用
cost_rate = 0.0015  # 单次交易成本约0.15%
# 买卖双向：0.3%
# 每周调仓一次，年化成本：0.3% * 52 = 15.6%
```

---

## 改进后的策略设计

### 策略框架
```
1. 大盘过滤（每日）
   ↓
2. 选股（每周一）
   - 选前5%（约25只）
   - 预测得分 > 阈值
   ↓
3. 持仓管理（动态）
   - 持有3-5天
   - 止损：-5%
   - 止盈：+10%
   - 预测得分转负时提前卖出
   ↓
4. 成本控制
   - 考虑滑点0.1%
   - 手续费0.03%
```

### 预期效果

| 指标 | 当前 | 目标 | 说明 |
|------|------|------|------|
| 交易频率 | 99笔/天 | 25笔/周 | 降低80% |
| 持仓时间 | 1天 | 3-5天 | 捕捉趋势 |
| 换手率 | 极高 | 适中 | 降低成本 |
| 大盘过滤 | 无 | 有 | 避开熊市 |
| 年化成本 | ~50% | ~15% | 成本可控 |

---

## 实施计划

### Week 1: 降低交易频率
- [ ] 修改选股比例（20% → 5%）
- [ ] 添加预测得分阈值过滤
- [ ] 回测验证

### Week 2: 延长持仓时间
- [ ] 实现3天持仓逻辑
- [ ] 添加止损止盈机制
- [ ] 回测验证

### Week 3: 大盘过滤
- [ ] 实现大盘过滤器
- [ ] 集成到回测框架
- [ ] 参数调优

### Week 4: 成本控制
- [ ] 添加滑点和手续费模型
- [ ] 综合回测
- [ ] 对比分析

---

## 相关链接

- 当前回测代码：run_backtest_fixed.py
- 交易明细：backtest_result/trades_csi500_2025_detailed.csv
- 模型文件：model/model_csi500.pkl
