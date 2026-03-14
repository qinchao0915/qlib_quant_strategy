---
date: 2026-03-13
type: model_issue
status: pending
severity: P1
---

## 问题/发现

模型保存时 `V7Trainer` 类没有 `self.model` 属性导致报错

### 详细描述
- 训练完成后保存模型时失败
- 错误：`'V7Trainer' object has no attribute 'model'`
- 原因是 `V7Trainer` 使用 `self.models` 字典而非 `self.model`

### 关键代码
```python
# 错误代码
pickle.dump({
    'model': trainer.model,  # ❌ 不存在
    ...
}, f)

# 修复代码
trainer.save(model_path, feature_cols, ic)  # ✅ 使用内置方法
```

## 根因分析

1. `V7Trainer` 类设计使用 `self.models` 存储多个模型
2. 保存代码假设有 `self.model` 单模型
3. 接口不一致导致 bug

## 解决方案

### 临时处理
- 修改保存代码，使用 `trainer.save()` 方法

### 长期方案
1. 统一模型接口，明确单模型 vs 多模型场景
2. 添加类型检查，防止类似错误
3. 建立模型保存/加载的单元测试

## 验证结果

- [x] 保存代码已修复
- [x] 模型可正常保存
- [x] 三个指数模型已保存成功

## 相关链接

- 修复文件：run_model_training.py
- 模型文件：model/model_*.pkl
