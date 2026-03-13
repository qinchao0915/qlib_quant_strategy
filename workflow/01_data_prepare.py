#!/usr/bin/env python3
"""
Qlib 数据准备脚本

功能：
1. 获取 CSI500 成分股列表
2. 下载日线价格数据
3. 保存到 data/ 目录供后续训练和回测使用

运行：
    python workflow/01_data_prepare.py
"""

import sys
import yaml
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tushare_provider.tushare_fetcher import TushareDataFetcher


def prepare_data():
    """准备训练数据"""
    print("=" * 60)
    print("📊 Qlib 数据准备")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent.parent / 'config' / 'workflow_config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # 创建数据目录
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    cache_dir = data_dir / "cache"
    
    for d in [data_dir, raw_dir, processed_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {d}")
    
    # 初始化数据获取器
    print("\n🔧 初始化 Tushare 数据获取器...")
    fetcher = TushareDataFetcher(
        cfg['data']['tushare_token'],
        str(cache_dir)
    )
    
    # 获取股票列表
    market = cfg['data']['market']
    print(f"\n📥 获取 {market.upper()} 成分股...")
    stocks = fetcher.get_stock_list(market)
    print(f"✅ 股票数量: {len(stocks)}")
    
    if not stocks:
        print("❌ 获取股票列表失败")
        return False
    
    # 保存股票列表
    stock_list_path = processed_dir / f"{market}_stocks.txt"
    with open(stock_list_path, 'w') as f:
        f.write('\n'.join(stocks))
    print(f"✅ 股票列表保存: {stock_list_path}")
    
    # 下载训练数据
    train_start = cfg['data']['train_start']
    train_end = cfg['data']['train_end']
    valid_start = cfg['data']['valid_start']
    valid_end = cfg['data']['valid_end']
    
    print(f"\n📥 下载训练数据 ({train_start} ~ {train_end})...")
    train_df = fetcher.get_daily_prices_batch(stocks, train_start, train_end)
    
    if not train_df.empty:
        train_path = raw_dir / f"{market}_train_{train_start}_{train_end}.csv"
        train_df.to_csv(train_path, index=False)
        print(f"✅ 训练数据: {len(train_df)} 条记录")
        print(f"✅ 保存到: {train_path}")
    else:
        print("❌ 训练数据为空")
        return False
    
    # 下载验证数据
    print(f"\n📥 下载验证数据 ({valid_start} ~ {valid_end})...")
    valid_df = fetcher.get_daily_prices_batch(stocks, valid_start, valid_end)
    
    if not valid_df.empty:
        valid_path = raw_dir / f"{market}_valid_{valid_start}_{valid_end}.csv"
        valid_df.to_csv(valid_path, index=False)
        print(f"✅ 验证数据: {len(valid_df)} 条记录")
        print(f"✅ 保存到: {valid_path}")
    else:
        print("⚠️ 验证数据为空（可能日期较新，数据未更新）")
    
    # 生成数据摘要
    print("\n" + "=" * 60)
    print("📈 数据摘要")
    print("=" * 60)
    print(f"股票池: {market.upper()}")
    print(f"股票数量: {len(stocks)}")
    print(f"训练数据: {len(train_df)} 条 ({train_start} ~ {train_end})")
    print(f"验证数据: {len(valid_df)} 条 ({valid_start} ~ {valid_end})")
    print(f"\n数据目录: {data_dir.absolute()}")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = prepare_data()
    sys.exit(0 if success else 1)
