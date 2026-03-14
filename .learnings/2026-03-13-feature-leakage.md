---
date: 2026-03-13
type: data_issue
status: pending
severity: P0
---

## 问题/发现

回测中发现 `return_1d` 特征导致数据穿越，收益率虚高

### 详细描述
- `return_1d` 是当日收益率，属于未来信息
- 模型用它预测次日收益，造成信息泄露
- CSI500 回测年化收益从 228% 修正为 85%

### 数据样本
```python
# 错误用法 - 使用了当日收益
avg_return = top_stocks['close'].mean() / top_stocks['pre_close'].mean() - 1

# 正确用法 - 使用次日收益
avg_return = (top_stocks['next_close'].mean() / top_stocks['next_open'].mean()) - 1
```

## 根因分析

1. 特征工程中 `return_1d = close.pct_change(1)` 包含未来信息
2. 回测逻辑错误，使用了 T+0 收益而非 T+1 收益
3. 未做数据穿越检查

## 解决方案

### 临时处理
- 修复回测代码，使用 T+1 收益计算
- 重新评估模型表现

### 长期方案
1. 建立特征审查清单，标记潜在的未来特征
2. 回测前强制做数据穿越检查
3. 添加单元测试验证回测逻辑

## 验证结果

- [x] 回测代码已修复
- [x] T+1 收益率计算正确
- [x] 修正后年化收益 85%（更合理）
- [ ] 特征审查清单已建立

## 相关链接

- 修复代码：run_backtest_fixed.py
- 原问题代码：run_backtest.py
- 分析报告：backtest_result/
