# Qlib Quant 学习记录目录

> 量化项目的经验沉淀与知识管理

---

## 目录结构

```
.learnings/
├── README.md              # 本文件
├── archived/              # 已归档条目
├── template/              # 记录模板
│   ├── data_issue.md      # 数据问题模板
│   ├── model_issue.md     # 模型问题模板
│   └── improvement.md     # 改进点模板
└── YYYY-MM-DD-*.md        # 具体学习记录
```

---

## 记录类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `data_issue` | 数据相关问题 | 数据穿越、缺失值、异常值 |
| `model_issue` | 模型相关问题 | 过拟合、欠拟合、训练失败 |
| `improvement` | 优化改进 | 特征优化、模型调优、流程改进 |
| `insight` | 新发现/洞察 | 市场规律、特征重要性发现 |

---

## 严重程度分级

| 级别 | 说明 | 响应时间 |
|------|------|----------|
| P0 | 严重问题，影响实盘 | 立即处理 |
| P1 | 重要问题，影响回测 | 24小时内 |
| P2 | 一般问题，优化建议 | 本周内 |

---

## 记录流程

### 1. 发现问题时
```bash
# 复制模板
cp .learnings/template/data_issue.md .learnings/2026-03-14-问题描述.md

# 填写内容
# ...

# 设置状态为 pending
```

### 2. 每日反思（22:00）
- 检查所有 `status: pending` 的条目
- 评估是否 promote 到 MEMORY.md
- 更新状态

### 3. 晋升到 MEMORY.md
- 有价值的经验更新到 MEMORY.md
- 原文件状态改为 `promoted`
- 可选：移动到 archived/

---

## 现有记录

### 待处理 (pending)
- [2026-03-13-feature-leakage.md](2026-03-13-feature-leakage.md) - P0 - 数据穿越问题
- [2026-03-13-model-save-bug.md](2026-03-13-model-save-bug.md) - P1 - 模型保存bug

### 已归档 (archived)
暂无

---

## 快速开始

```bash
# 创建新的学习记录
cp .learnings/template/data_issue.md .learnings/$(date +%Y-%m-%d)-问题描述.md

# 编辑记录
vim .learnings/$(date +%Y-%m-%d)-问题描述.md

# 提交到 git
git add .learnings/
git commit -m "Add learning record: 问题描述"
```

---

## 关联文档

- [MEMORY.md](../MEMORY.md) - 长期记忆
- [AGENTS.md](../AGENTS.md) - Agent 规范
- [SOUL.md](../SOUL.md) - 核心身份
