import argparse
import sys
import os
import pandas as pd
import logging
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
from src.analysis.results_handler import store_analysis_results
from src.utils.ticker_utils import get_company_name

def setup_logging(stock_dir):
    log_file = os.path.join(stock_dir, 'run.log')
    # Clear any existing handlers to prevent duplicate logging
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also keep console output
        ],
        force=True  # Force reconfiguration
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Stock trading strategy analysis')
    parser.add_argument('--ticker', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--stocklist', help='CSV file containing stock tickers, names, and sectors')
    parser.add_argument('--tradecost', type=float, default=0.15, help='Trading cost percentage (default: 0.15)')
    parser.add_argument('--stoploss', type=float, default=-5, help='Stop loss percentage (default: -5)')
    parser.add_argument('--trigger-resolution', type=float, default=0.5, help='Resolution for trigger analysis')
    parser.add_argument('--max-buytrigger', type=float, default=5, help='Maximum buy trigger percentage')
    parser.add_argument('--max-selltrigger', type=float, default=5, help='Maximum sell trigger percentage')
    
    global args
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if args.ticker and args.stocklist:
        logging.error("Error: Please provide either a ticker or a stocklist, not both.")
        sys.exit(1)
    
    if args.ticker:
        stock = args.ticker
        stock_dir = os.path.join('results', stock)
        ensure_stock_dir(stock)
        logger = setup_logging(stock_dir)
        
        logger.info(f"\nProcessing {stock}...")
        
        # Load or fetch data
        data = load_from_csv(stock)
        if data is None:
            logger.info(f"\nFetching data for {stock} from Yahoo Finance...")
            data = get_daily_data(stock)
            save_to_csv(data, stock)
        else:
            logger.info(f"\nData for {stock} already exists. Loading from CSV...")
        
        validate_data(data, stock)
        
        # Run strategies
        logger.info("\nRunning DCA Strategy...")
        dca_trades, dca_final, dca_values = run_monthly_dca_strategy(data, tradecost=args.tradecost)
        logger.info(f"Final Portfolio Value: ${dca_final:.2f}")
        logger.info(f"Max Drawdown: {calculate_max_drawdown(dca_values['value']):.1f}%")
        logger.info("\nDCA Strategy Trades:")
        for _, trade in dca_trades.iterrows():
            logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['invested']:.2f}")
        
        logger.info("\nRunning Buy & Hold Strategy...")
        bh_trades, bh_final, bh_values = run_buy_and_hold_strategy(data, tradecost=args.tradecost)
        logger.info(f"Final Portfolio Value: ${bh_final:.2f}")
        logger.info(f"Max Drawdown: {calculate_max_drawdown(bh_values['value']):.1f}%")
        logger.info("\nBuy & Hold Strategy Trades:")
        for _, trade in bh_trades.iterrows():
            logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
        
        # Run 3D analysis
        logger.info("\nRunning 3D Analysis...")
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
        
        logger.info(f"\nBest Parameters Found:")
        logger.info(f"Buy Trigger: {best_buy:.1f}%")
        logger.info(f"Sell Trigger: {best_sell:.1f}%")
        logger.info(f"Days Window: {int(best_days)}")
        logger.info(f"Final Value: ${best_value:.2f}")
        
        # Find worst parameters
        worst_idx = results[:, 2].argmin()
        worst_buy = results[worst_idx, 0]
        worst_sell = results[worst_idx, 1]
        worst_days = results[worst_idx, 3]
        worst_value = results[worst_idx, 2]
        
        logger.info(f"\nWorst Parameters Found:")
        logger.info(f"Buy Trigger: {worst_buy:.1f}%")
        logger.info(f"Sell Trigger: {worst_sell:.1f}%")
        logger.info(f"Days Window: {int(worst_days)}")
        logger.info(f"Final Value: ${worst_value:.2f}")
        
        # Run best dip strategy
        logger.info("\nRunning Best Dip Strategy...")
        optimized_trades, optimized_final, optimized_values = run_dip_recovery_strategy(
            data, 
            buy_trigger=best_buy,
            sell_trigger=best_sell,
            days_window=int(best_days),
            tradecost=args.tradecost
        )
        logger.info("\nBest Dip Strategy Trades:")
        for _, trade in optimized_trades.iterrows():
            if trade['action'].lower() == 'buy':
                logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
            else:
                profit_str = f"${trade['profit']:.2f}" if not pd.isna(trade['profit']) else "N/A"
                return_str = f"{trade['return']:.1f}%" if not pd.isna(trade['return']) else "N/A"
                logger.info(f"SELL: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Return: {return_str}, Profit: {profit_str}")
        
        # Run worst dip strategy
        logger.info("\nRunning Worst Dip Strategy...")
        worst_trades, worst_final, worst_values = run_dip_recovery_strategy(
            data,
            buy_trigger=worst_buy,
            sell_trigger=worst_sell,
            days_window=int(worst_days),
            tradecost=args.tradecost
        )
        logger.info("\nWorst Dip Strategy Trades:")
        for _, trade in worst_trades.iterrows():
            if trade['action'].lower() == 'buy':
                logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
            else:
                profit_str = f"${trade['profit']:.2f}" if not pd.isna(trade['profit']) else "N/A"
                return_str = f"{trade['return']:.1f}%" if not pd.isna(trade['return']) else "N/A"
                logger.info(f"SELL: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Return: {return_str}, Profit: {profit_str}")
        
        # Create plots
        logger.info("\nGenerating plots...")
        company_name = get_company_name(stock)
        create_3d_strategy_plot(
            results, bh_final, dca_final,
            stock, args.stoploss, args.tradecost, company_name
        )
            
        create_strategy_comparison_plot(
            optimized_values, worst_values, dca_values, bh_values,
            optimized_trades, dca_trades,
            stock, args.stoploss, args.tradecost,
            int(best_days), best_buy, best_sell, company_name
        )
        # Store analysis results
        company_name = get_company_name(stock)
        store_analysis_results(args, dca_final, dca_values, optimized_final, optimized_values, bh_final, bh_values, 
                              best_value, best_buy, best_sell, best_days,
                              worst_value, worst_buy, worst_sell, worst_days,
                              optimized_trades, worst_trades, bh_trades, dca_trades, 
                              stock, company_name)
        logger.info(f"\nAnalysis complete. Check the 'results/{stock}' directory for visualizations and data.")
    elif args.stocklist:
        # Initialize logging to results directory for stocklist overview
        logger = setup_logging('results')
        logger.info(f"\nLoading stock list from {args.stocklist}...")
        import csv
        try:
            with open(args.stocklist, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader) # Skip header row
                tickers = [row[0] for row in csv_reader]
        except FileNotFoundError:
            logger.error(f"Error: Could not find file {args.stocklist}")
            sys.exit(1)
        except Exception as e:
             logger.error(f"Error: Could not load stock list from {args.stocklist}. {e}")
             sys.exit(1)
        
        for stock in tickers:
            stock_dir = os.path.join('results', stock)
            ensure_stock_dir(stock)
            # Setup logging for this specific stock
            logger = setup_logging(stock_dir)
            
            logger.info(f"\nProcessing {stock}...")
            
            # Load or fetch data
            data = load_from_csv(stock)
            if data is None:
                logger.info(f"\nFetching data for {stock} from Yahoo Finance...")
                data = get_daily_data(stock)
                save_to_csv(data, stock)
            else:
                logger.info(f"\nData for {stock} already exists. Loading from CSV...")
            
            validate_data(data, stock)
            
            # Run strategies
            logger.info("\nRunning DCA Strategy...")
            dca_trades, dca_final, dca_values = run_monthly_dca_strategy(data, tradecost=args.tradecost)
            logger.info(f"Final Portfolio Value: ${dca_final:.2f}")
            logger.info(f"Max Drawdown: {calculate_max_drawdown(dca_values['value']):.1f}%")
            logger.info("\nDCA Strategy Trades:")
            for _, trade in dca_trades.iterrows():
                logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['invested']:.2f}")
            
            logger.info("\nRunning Buy & Hold Strategy...")
            bh_trades, bh_final, bh_values = run_buy_and_hold_strategy(data, tradecost=args.tradecost)
            logger.info(f"Final Portfolio Value: ${bh_final:.2f}")
            logger.info(f"Max Drawdown: {calculate_max_drawdown(bh_values['value']):.1f}%")
            logger.info("\nBuy & Hold Strategy Trades:")
            for _, trade in bh_trades.iterrows():
                logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
            
            # Run 3D analysis
            logger.info("\nRunning 3D Analysis...")
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
            
            logger.info(f"\nBest Parameters Found:")
            logger.info(f"Buy Trigger: {best_buy:.1f}%")
            logger.info(f"Sell Trigger: {best_sell:.1f}%")
            logger.info(f"Days Window: {int(best_days)}")
            logger.info(f"Final Value: ${best_value:.2f}")
            
            # Find worst parameters
            worst_idx = results[:, 2].argmin()
            worst_buy = results[worst_idx, 0]
            worst_sell = results[worst_idx, 1]
            worst_days = results[worst_idx, 3]
            worst_value = results[worst_idx, 2]
            
            logger.info(f"\nWorst Parameters Found:")
            logger.info(f"Buy Trigger: {worst_buy:.1f}%")
            logger.info(f"Sell Trigger: {worst_sell:.1f}%")
            logger.info(f"Days Window: {int(worst_days)}")
            logger.info(f"Final Value: ${worst_value:.2f}")
            
            # Run best dip strategy
            logger.info("\nRunning Best Dip Strategy...")
            optimized_trades, optimized_final, optimized_values = run_dip_recovery_strategy(
                data, 
                buy_trigger=best_buy,
                sell_trigger=best_sell,
                days_window=int(best_days),
                tradecost=args.tradecost
            )
            logger.info("\nBest Dip Strategy Trades:")
            for _, trade in optimized_trades.iterrows():
                if trade['action'].lower() == 'buy':
                    logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
                else:
                    profit_str = f"${trade['profit']:.2f}" if not pd.isna(trade['profit']) else "N/A"
                    return_str = f"{trade['return']:.1f}%" if not pd.isna(trade['return']) else "N/A"
                    logger.info(f"SELL: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Return: {return_str}, Profit: {profit_str}")
            
            # Run worst dip strategy
            logger.info("\nRunning Worst Dip Strategy...")
            worst_trades, worst_final, worst_values = run_dip_recovery_strategy(
                data,
                buy_trigger=worst_buy,
                sell_trigger=worst_sell,
                days_window=int(worst_days),
                tradecost=args.tradecost
            )
            logger.info("\nWorst Dip Strategy Trades:")
            for _, trade in worst_trades.iterrows():
                if trade['action'].lower() == 'buy':
                    logger.info(f"BUY: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Cost: ${trade['trading_cost']:.2f}, Total: ${trade['value']:.2f}")
                else:
                    profit_str = f"${trade['profit']:.2f}" if not pd.isna(trade['profit']) else "N/A"
                    return_str = f"{trade['return']:.1f}%" if not pd.isna(trade['return']) else "N/A"
                    logger.info(f"SELL: {trade['date'].strftime('%Y-%m-%d')}, Price: ${trade['price']:.2f}, Shares: {trade['shares']:.2f}, Return: {return_str}, Profit: {profit_str}")
            
            # Create plots
            logger.info("\nGenerating plots...")
            company_name = get_company_name(stock)
            create_3d_strategy_plot(
                results, bh_final, dca_final,
                stock, args.stoploss, args.tradecost, company_name
            )
            
            create_strategy_comparison_plot(
                optimized_values, worst_values, dca_values, bh_values,
                optimized_trades, dca_trades,
                stock, args.stoploss, args.tradecost,
                int(best_days), best_buy, best_sell, company_name
            )
            
            # Store analysis results
            company_name = get_company_name(stock)
            store_analysis_results(args, dca_final, dca_values, optimized_final, optimized_values, bh_final, bh_values, 
                                  best_value, best_buy, best_sell, best_days,
                                  worst_value, worst_buy, worst_sell, worst_days,
                                  optimized_trades, worst_trades, bh_trades, dca_trades, 
                                  stock, company_name)
            logger.info(f"\nAnalysis complete. Check the 'results/' directory for visualizations and data.")
    
    else:
        logging.error("Please provide a ticker using --ticker or a stock list using --stocklist")
        sys.exit(1)

if __name__ == "__main__":
    main()
