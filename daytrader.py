import argparse
import sys
import os
from src.utils.data_handlers import (
    get_daily_data,
    validate_data,
    get_csv_filename,
    load_from_csv,
    save_to_csv,
    store_facts_to_file,
    ensure_stock_dir
)
from src.strategies.trading_strategies import (
    run_monthly_dca_strategy,
    run_dip_recovery_strategy,
    run_buy_and_hold_strategy
)
from src.analysis.strategy_analysis import (
    calculate_max_drawdown,
    run_strategy_with_parameters,
    create_3d_analysis
)
from src.visualization.plotting import (
    create_3d_strategy_plot,
    create_strategy_comparison_plot
)

def main():
    parser = argparse.ArgumentParser(description='Stock trading strategy analysis')
    parser.add_argument('stock', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--tradecost', type=float, default=0.15, help='Trading cost percentage (default: 0.15)')
    parser.add_argument('--stoploss', type=float, default=-5, help='Stop loss percentage (default: -5)')
    parser.add_argument('--trigger-resolution', type=float, default=0.5, help='Resolution for trigger analysis')
    parser.add_argument('--max-buytrigger', type=float, default=5, help='Maximum buy trigger percentage')
    parser.add_argument('--max-selltrigger', type=float, default=5, help='Maximum sell trigger percentage')
    
    global args
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create stock-specific results directory
    ensure_stock_dir(args.stock)
    
    # Load or fetch data
    data = load_from_csv(args.stock)
    if data is None:
        data = get_daily_data(args.stock)
        save_to_csv(data, args.stock)
    
    validate_data(data, args.stock)
    
    # Run strategies
    print("\nRunning DCA Strategy...")
    dca_trades, dca_final, dca_values = run_monthly_dca_strategy(data, tradecost=args.tradecost)
    print(f"Final Portfolio Value: ${dca_final:.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(dca_values['value']):.1f}%")
    
    print("\nRunning Dip Recovery Strategy...")
    dip_trades, dip_final, dip_values = run_dip_recovery_strategy(data, tradecost=args.tradecost)
    print(f"Final Portfolio Value: ${dip_final:.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(dip_values['value']):.1f}%")
    
    print("\nRunning Buy & Hold Strategy...")
    bh_trades, bh_final, bh_values = run_buy_and_hold_strategy(data, tradecost=args.tradecost)
    print(f"Final Portfolio Value: ${bh_final:.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(bh_values['value']):.1f}%")
    
    # Run 3D analysis
    print("\nRunning 3D Analysis...")
    results = create_3d_analysis(
        data,
        trigger_resolution=args.trigger_resolution,
        max_buytrigger=args.max_buytrigger,
        max_selltrigger=args.max_selltrigger
    )
    
    # Find best parameters
    best_idx = results[:, 2].argmax()
    best_buy = results[best_idx, 0]
    best_sell = results[best_idx, 1]
    best_days = results[best_idx, 3]
    best_value = results[best_idx, 2]
    
    print(f"\nBest Parameters Found:")
    print(f"Buy Trigger: {best_buy:.1f}%")
    print(f"Sell Trigger: {best_sell:.1f}%")
    print(f"Days Window: {int(best_days)}")
    print(f"Final Value: ${best_value:.2f}")
    
    # Find worst parameters
    worst_idx = results[:, 2].argmin()
    worst_buy = results[worst_idx, 0]
    worst_sell = results[worst_idx, 1]
    worst_days = results[worst_idx, 3]
    worst_value = results[worst_idx, 2]
    
    # Run optimized strategy
    print("\nRunning Optimized Strategy...")
    optimized_trades, optimized_final, optimized_values = run_dip_recovery_strategy(
        data, tradecost=args.tradecost
    )
    
    # Run worst strategy
    print("\nRunning Worst Strategy...")
    worst_trades, worst_final, worst_values = run_dip_recovery_strategy(
        data, tradecost=args.tradecost
    )
    
    # Create plots
    print("\nGenerating plots...")
    create_3d_strategy_plot(
        results, bh_final, dca_final,
        args.stock, args.stoploss, args.tradecost
    )
    
    create_strategy_comparison_plot(
        optimized_values, worst_values, dca_values, bh_values,
        optimized_trades, dca_trades,
        args.stock, args.stoploss, args.tradecost,
        int(best_days), best_buy, best_sell
    )
    
    # Store analysis results
    facts = {
        "parameters": vars(args),
        "strategy_results": {
            "dca": {
                "final_value": dca_final,
                "max_drawdown": calculate_max_drawdown(dca_values['value']),
                "trades": len(dca_trades)
            },
            "dip_recovery": {
                "final_value": dip_final,
                "max_drawdown": calculate_max_drawdown(dip_values['value']),
                "trades": len(dip_trades)
            },
            "buy_and_hold": {
                "final_value": bh_final,
                "max_drawdown": calculate_max_drawdown(bh_values['value']),
                "trades": len(bh_trades)
            },
            "optimized": {
                "final_value": best_value,
                "buy_trigger": best_buy,
                "sell_trigger": best_sell,
                "days_window": int(best_days)
            }
        }
    }
    store_facts_to_file(facts, args.stock)
    print(f"\nAnalysis complete. Check the 'results/{args.stock}' directory for visualizations and data.")

if __name__ == "__main__":
    main()
