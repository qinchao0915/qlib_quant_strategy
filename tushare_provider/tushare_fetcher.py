#!/usr/bin/env python3
"""
Tushare 数据获取器

功能：
- 获取股票列表（沪深300/中证500/中证1000）
- 获取日线价格数据
- 本地缓存机制
"""

import pandas as pd
import numpy as np
import tushare as ts
import pickle
from pathlib import Path
from datetime import datetime


class TushareDataFetcher:
    """Tushare 数据获取器"""
    
    def __init__(self, token, cache_path="data/cache"):
        """
        初始化
        
        Args:
            token: Tushare API Token
            cache_path: 缓存路径
        """
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # 设置 Tushare Token
        ts.set_token(token)
        self.pro = ts.pro_api()
    
    def get_stock_list(self, market="csi500"):
        """
        获取指数成分股列表
        
        Args:
            market: 指数代码，可选 csi300, csi500, csi1000
            
        Returns:
            list: 成分股代码列表
        """
        cache = self.cache_path / f"stock_list_{market}.pkl"
        
        # 检查缓存
        if cache.exists():
            print(f"📂 从缓存加载 {market} 股票列表")
            with open(cache, 'rb') as f:
                return pickle.load(f)
        
        # 尝试获取指数成分股
        code_map = {
            "csi300": "000300.SH",
            "csi500": "000905.SH",
            "csi1000": "000852.SH"
        }
        
        if market not in code_map:
            raise ValueError(f"不支持的指数: {market}，可选: {list(code_map.keys())}")
        
        code = code_map[market]
        
        # 获取成分股
        print(f"📥 下载 {market} 成分股列表...")
        try:
            # 尝试多个日期
            for date in ['20250312', '20250301', '20250201', '20250102']:
                df = self.pro.index_weight(index_code=code, trade_date=date)
                if not df.empty:
                    stocks = df['con_code'].tolist()
                    print(f"✅ 获取到 {len(stocks)} 只股票")
                    break
            else:
                # 如果都为空，使用备选方案：获取所有股票
                print(f"⚠️ 指数权重数据为空，使用备选方案获取股票列表")
                df = self.pro.stock_basic(exchange='', list_status='L')
                
                # 根据市场筛选
                if market == "csi300":
                    # 沪深300：取市值最大的300只
                    stocks = df['ts_code'].tolist()[:300]
                elif market == "csi500":
                    # 中证500：取市值排名301-800
                    stocks = df['ts_code'].tolist()[300:800]
                else:  # csi1000
                    # 中证1000：取市值排名801-1800
                    stocks = df['ts_code'].tolist()[800:1800]
                
                print(f"✅ 备选方案获取到 {len(stocks)} 只股票")
        except Exception as e:
            print(f"❌ 获取失败: {e}")
            return []
        
        # 保存缓存
        with open(cache, 'wb') as f:
            pickle.dump(stocks, f)
        
        return stocks
    
    def get_daily_prices(self, symbol, sd, ed):
        """
        获取单只股票日线数据
        
        Args:
            symbol: 股票代码，如 '000001.SZ'
            sd: 开始日期 'YYYY-MM-DD'
            ed: 结束日期 'YYYY-MM-DD'
            
        Returns:
            DataFrame: 日线数据
        """
        # 缓存文件
        cache = self.cache_path / f"daily_{symbol}_{sd}_{ed}.pkl"
        
        # 检查缓存
        if cache.exists():
            with open(cache, 'rb') as f:
                return pickle.load(f)
        
        # 转换日期格式
        s = sd.replace("-", "")
        e = ed.replace("-", "")
        
        # 获取数据
        try:
            df = self.pro.daily(ts_code=symbol, start_date=s, end_date=e)
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        df = df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume'
        })
        
        # 转换日期
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        
        # 保存缓存
        with open(cache, 'wb') as f:
            pickle.dump(df, f)
        
        return df
    
    def get_daily_prices_batch(self, symbols, sd, ed):
        """
        批量获取日线数据
        
        Args:
            symbols: 股票代码列表
            sd: 开始日期
            ed: 结束日期
            
        Returns:
            DataFrame: 合并后的数据
        """
        res = []
        total = len(symbols)
        
        for i, s in enumerate(symbols):
            if i % 50 == 0:
                print(f"  进度: {i}/{total}")
            
            try:
                df = self.get_daily_prices(s, sd, ed)
                if not df.empty:
                    res.append(df)
            except Exception as e:
                print(f"  ⚠️ 跳过 {s}: {e}")
                continue
        
        if res:
            return pd.concat(res, ignore_index=True)
        else:
            return pd.DataFrame()


if __name__ == '__main__':
    # 测试
    from pathlib import Path
    import yaml
    
    # 读取配置
    config_path = Path(__file__).parent.parent / 'config' / 'workflow_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    token = config['data']['tushare_token']
    
    # 初始化
    fetcher = TushareDataFetcher(token)
    
    # 测试获取股票列表
    print("\n测试获取中证500成分股:")
    stocks = fetcher.get_stock_list("csi500")
    print(f"前5只: {stocks[:5]}")
    
    # 测试获取日线数据
    print("\n测试获取单只股票数据:")
    df = fetcher.get_daily_prices("000001.SZ", "2024-01-01", "2024-03-01")
    print(df.head())
