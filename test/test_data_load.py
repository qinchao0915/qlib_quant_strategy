#!/usr/bin/env python3
"""
数据加载测试

测试 DataLoader 是否能正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import DataLoader


def test_data_loader():
    """测试数据加载器"""
    print("=" * 60)
    print("🧪 测试 DataLoader")
    print("=" * 60)
    
    # 测试 1: 初始化
    print("\n测试 1: 初始化 DataLoader...")
    try:
        loader = DataLoader()
        print("✅ DataLoader 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 测试 2: 获取股票列表
    print("\n测试 2: 获取股票列表...")
    try:
        stocks = loader.get_stock_list(exchange='SZSE')
        print(f"✅ 获取到 {len(stocks)} 只深圳股票")
        print(f"  示例: {stocks['ts_code'].iloc[0]} - {stocks['name'].iloc[0]}")
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return False
    
    # 测试 3: 获取日线数据
    print("\n测试 3: 获取日线数据（平安银行 000001.SZ）...")
    try:
        df = loader.get_daily_data('000001.SZ', '20240301', '20240312')
        if not df.empty:
            print(f"✅ 获取到 {len(df)} 条数据")
            print(f"  日期范围: {df.index[0]} ~ {df.index[-1]}")
            print(f"  收盘价: {df['close'].iloc[-1]:.2f}")
        else:
            print("⚠️  数据为空")
    except Exception as e:
        print(f"❌ 获取日线数据失败: {e}")
        return False
    
    # 测试 4: 转换为 Qlib 格式
    print("\n测试 4: 转换为 Qlib 格式...")
    try:
        qlib_df = loader.to_qlib_format(df)
        print(f"✅ 转换成功")
        print(f"  列名: {list(qlib_df.columns)}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_data_loader()
    sys.exit(0 if success else 1)
