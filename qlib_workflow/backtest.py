#!/usr/bin/env python3
"""
Qlib Backtest 组件

功能：
- 回测引擎
- 绩效分析
- 可视化
"""

import pandas as pd
import numpy as np


class QlibBacktest:
    """Qlib 风格回测"""
    
    def __init__(self, start_date, end_date, benchmark='000905.SH'):
        """
        初始化
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            benchmark: 基准指数
        """
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.results = None
    
    def run(self, signals, price_df):
        """
        运行回测
        
        Args:
            signals: 交易信号
            price_df: 价格数据
            
        Returns:
            dict: 回测结果
        """
        print("🔄 运行回测...")
        
        # 简化回测：等权持有选股
        daily_returns = []
        
        for date, group in signals.groupby(level='date'):
            if date < self.start_date or date > self.end_date:
                continue
            
            # 获取当日选股
            selected = group[group['signal'] == 1]
            
            if len(selected) == 0:
                daily_return = 0
            else:
                # 获取次日收益
                symbols = selected.index.get_level_values('symbol').tolist()
                # 简化：假设持有1天
                daily_return = 0  # 需要实际价格数据计算
            
            daily_returns.append({
                'date': date,
                'return': daily_return
            })
        
        self.results = pd.DataFrame(daily_returns)
        
        # 计算指标
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _calculate_metrics(self):
        """计算回测指标"""
        if self.results is None or self.results.empty:
            return {}
        
        returns = self.results['return']
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': (1 + returns).prod() ** (252 / len(returns)) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'volatility': returns.std() * np.sqrt(252),
            'win_rate': (returns > 0).mean()
        }
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(returns):
        """计算最大回撤"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def report(self):
        """生成回测报告"""
        if self.results is None:
            print("❌ 请先运行回测")
            return
        
        print("\n" + "=" * 60)
        print("📊 回测报告")
        print("=" * 60)
        
        metrics = self._calculate_metrics()
        
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("=" * 60)
