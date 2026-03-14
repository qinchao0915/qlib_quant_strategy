import pandas as pd
import os

def validate_backtest_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        print("请检查路径是否正确。")
        return

    print(f"📊 开始分析回测结果文件: {csv_path}")
    print("-" * 50)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 1. 基础统计
    total_trades = len(df)
    if total_trades == 0:
        print("⚠️ 交易记录为空！")
        return
        
    win_rate = (df['return'] > 0).mean() * 100
    avg_return = df['return'].mean() * 100
    avg_holding_days = df['holding_days'].mean()
    
    print(f"【基础指标】")
    print(f"总交易笔数: {total_trades}")
    print(f"总体胜率:   {win_rate:.2f}%")
    print(f"平均单笔收益: {avg_return:.2f}%")
    print(f"平均持仓天数: {avg_holding_days:.2f}天\n")
    
    # 2. 卖出原因分布分析 (核心Bug检查)
    print(f"【卖出原因分布与 Bug 验证】")
    if 'sell_signal' in df.columns:
        # 分组统计
        sell_stats = df.groupby('sell_signal').agg(
            笔数=('symbol', 'count'),
            平均收益=('return', lambda x: x.mean() * 100),
            平均持仓=('holding_days', 'mean')
        )
        sell_stats['占比(%)'] = (sell_stats['笔数'] / total_trades) * 100
        sell_stats = sell_stats[['笔数', '占比(%)', '平均收益', '平均持仓']]
        
        print(sell_stats.round(2).to_string())
        print("\n" + "="*20 + " 自动诊断报告 " + "="*20)
        
        # 诊断 1：换仓缓冲逻辑 Bug
        rank_drop_trades = df[df['sell_signal'].str.contains('排名下降|换仓', na=False)]
        if len(rank_drop_trades) == 0:
            print("❌ [严重问题] '排名下降/换仓' 卖出笔数依然为 0 ！")
            print("   👉 诊断：AI 写的持仓缓冲逻辑没有生效，代码依然在强制执行‘到期全卖’。")
        else:
            print(f"✅ [修复成功] '排名下降/换仓' 已正常触发，共 {len(rank_drop_trades)} 笔。缓冲逻辑生效。")
            
        # 诊断 2：持仓天数强制要求
        expire_trades = df[df['sell_signal'].str.contains('持仓到期', na=False)]
        if len(expire_trades) > 0:
            invalid_expire = expire_trades[expire_trades['holding_days'] < 5]
            if len(invalid_expire) > 0:
                print(f"❌ [逻辑漏洞] 发现 {len(invalid_expire)} 笔 '持仓到期' 交易持仓不足 5 天！")
            else:
                print("✅ [修复成功] 所有 '持仓到期' 的交易均严格满足 >= 5 天的条件。")
                
        # 诊断 3：ATR 止损效果评估
        stop_loss_trades = df[df['sell_signal'].str.contains('止损', na=False)]
        if len(stop_loss_trades) > 0:
            sl_ratio = len(stop_loss_trades) / total_trades
            sl_avg_ret = stop_loss_trades['return'].mean() * 100
            print(f"ℹ️  [止损评估] 触发率: {sl_ratio*100:.2f}% | 平均止损亏损: {sl_avg_ret:.2f}%")
            if sl_ratio > 0.15:
                print("   👉 提示：止损触发率依然偏高(>15%)，可能 ATR 乘数设置得太窄，建议从 3 倍放大到 4 倍。")
    else:
        print("⚠️ 找不到 'sell_signal' 列，无法分析详细卖出原因。")
        
    print("-" * 50)

if __name__ == "__main__":
    # 使用时，让 AI 把这里改成最新跑出来的 CSV 结果路径
    target_csv = "result/backtest_v3_optimized.csv" # 使用最新的result文件
    validate_backtest_results(target_csv)
