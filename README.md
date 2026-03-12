# Qlib 量化交易项目

> 基于微软 Qlib 框架的 A股量化策略系统

## 📁 项目结构

```
qlib_quant/
├── backtest_result/      # 回测结果存储
│   └── README.md
├── strategy_docs/        # 策略文档和模型说明
│   └── README.md
├── data/                 # 训练和回测数据
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── qlib_data/        # Qlib格式数据
├── model/                # 训练好的模型文件 (.pkl)
│   └── README.md
├── selected_stocks/      # 每日选股推荐 (CSV)
│   └── README.md
├── test/                 # 测试文件
│   └── test_data_load.py
├── workflow/             # 主程序
│   ├── 01_data_prepare.py      # 数据准备
│   ├── 02_feature_engineering.py # 特征工程
│   ├── 03_model_train.py       # 模型训练
│   ├── 04_backtest.py          # 回测
│   ├── 05_select_stocks.py     # 选股生成
│   └── 06_daily_workflow.py    # 每日自动流程
├── todo/                 # 计划和对话记录
│   ├── plan.md           # 项目计划
│   └── ai_dialogue.md    # AI对话记录
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── data_loader.py    # 数据加载
│   ├── feature_utils.py  # 特征工具
│   ├── model_utils.py    # 模型工具
│   └── trading_utils.py  # 交易工具
└── README.md             # 本文件
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys
在 `~/.bashrc` 或项目根目录 `.env` 中添加：
```bash
export TUSHARE_TOKEN="你的Tushare Token"
```

### 3. 数据准备
```bash
python workflow/01_data_prepare.py
```

### 4. 训练模型
```bash
python workflow/03_model_train.py
```

### 5. 生成选股
```bash
python workflow/05_select_stocks.py
```

## 📊 每日工作流程

```bash
# 自动执行完整流程
python workflow/06_daily_workflow.py
```

选股结果将保存在 `selected_stocks/YYYY-MM-DD_recommendation.csv`

## 📝 重要文件说明

| 文件/文件夹 | 用途 |
|------------|------|
| `selected_stocks/` | 每日选股推荐，CSV格式，用于实盘交易 |
| `strategy_docs/` | 存放模型文档、策略说明、训练记录 |
| `todo/plan.md` | 下一步计划、待办事项 |
| `todo/ai_dialogue.md` | 与AI对话记录、决策过程 |
| `utils/` | 共用方法，避免代码重复 |

## ⚠️ 注意事项

1. **API Keys 不要上传到 Git**
2. **模型文件较大，用 Git LFS 管理**
3. **回测结果定期清理，避免磁盘占满**
4. **实盘交易前务必充分回测**

## 📅 更新记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-03-12 | 项目初始化 | 创建项目结构 |

---

_Created by 海狗 🦭_
