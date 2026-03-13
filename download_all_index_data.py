#!/usr/bin/env python3
"""
下载 CSI300、CSI500、CSI1000 三个指数的数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tushare_provider.tushare_fetcher import TushareDataFetcher
import yaml


def download_index_data(fetcher, market, start_date, end_date):
    """下载单个指数的数据"""
    print(f"\n{'='*60}")
    print(f"📥 下载 {market.upper()} 数据")
    print(f"{'='*60}")
    
    # 获取股票列表
    print(f"\n1. 获取 {market} 成分股...")
    stocks = fetcher.get_stock_list(market)
    print(f"✅ {market} 成分股: {len(stocks)} 只")
    
    # 获取日线数据
    print(f"\n2. 获取日线数据 ({start_date} ~ {end_date})...")
    print(f"   股票数量: {len(stocks)}")
    print(f"   预计时间: {len(stocks) * 0.5:.0f} 秒 ~ {len(stocks) * 2:.0f} 秒")
    
    price_df = fetcher.get_daily_prices_batch(stocks, start_date, end_date)
    
    if not price_df.empty:
        print(f"✅ 获取到 {len(price_df)} 条记录")
        print(f"   股票数: {price_df['symbol'].nunique()}")
        print(f"   日期范围: {price_df['date'].min()} ~ {price_df['date'].max()}")
    else:
        print(f"❌ 数据为空")
    
    return stocks, price_df


def main():
    """主函数"""
    print("="*60)
    print("🚀 下载三大指数数据")
    print("="*60)
    
    # 加载配置
    with open("config/workflow_config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # 初始化
    fetcher = TushareDataFetcher(
        cfg['data']['tushare_token'],
        cache_path='data/cache'
    )
    
    start_date = cfg['data']['train_start']  # 2020-01-01
    end_date = cfg['data']['valid_end']      # 2025-08-31
    
    print(f"\n时间范围: {start_date} ~ {end_date}")
    
    # 下载三个指数 (CSI300, CSI500, CSI1000)
    indices = ['csi300', 'csi500', 'csi1000']
    results = {}
    
    for market in indices:
        stocks, price_df = download_index_data(fetcher, market, start_date, end_date)
        results[market] = {
            'stocks': len(stocks),
            'records': len(price_df),
            'symbols': price_df['symbol'].nunique() if not price_df.empty else 0
        }
    
    # 汇总
    print(f"\n{'='*60}")
    print("📊 下载完成汇总")
    print(f"{'='*60}")
    
    for market, data in results.items():
        print(f"\n{market.upper()}:")
        print(f"  成分股: {data['stocks']} 只")
        print(f"  记录数: {data['records']} 条")
        print(f"  实际股票: {data['symbols']} 只")
    
    print(f"\n{'='*60}")
    print("✅ 所有数据下载完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
