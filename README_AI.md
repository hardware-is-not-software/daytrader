# DayTrader AI Guide

This document helps AI assistants understand the structure and functionality of the DayTrader codebase.

## Project Structure

```
daytrader/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_handlers.py      # Data loading and validation functions
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── trading_strategies.py # Trading strategy implementations
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── strategy_analysis.py  # Analysis and optimization functions
│   └── __init__.py
├── daytrader.py                  # Main entry point
└── README_AI.md                  # This file
```

## Module Functions

### utils/data_handlers.py
- `store_facts_to_file(facts, filename)`: Store analysis results in JSON format
- `save_to_csv(data, filename)`: Save data to CSV file
- `load_from_csv(filename)`: Load data from CSV file
- `get_csv_filename(ticker)`: Generate CSV filename for a ticker
- `get_daily_data(stock)`: Fetch daily stock data using yfinance
- `validate_data(data, stock)`: Validate stock data structure

### strategies/trading_strategies.py
- `run_monthly_dca_strategy(data, monthly_investment, max_investment, tradecost)`: Dollar-cost averaging strategy
- `run_dip_recovery_strategy(data, investment_amount, tradecost)`: Buy the dip strategy
- `run_buy_and_hold_strategy(data, investment_amount, tradecost)`: Simple buy and hold strategy

### analysis/strategy_analysis.py
- `calculate_max_drawdown(values)`: Calculate maximum portfolio drawdown
- `run_strategy_with_parameters(data, buy_trigger, sell_trigger, days_window, investment_amount, trading_cost)`: Test strategy with specific parameters
- `create_3d_analysis(data, trigger_resolution, max_buytrigger, max_selltrigger)`: Analyze strategy performance across parameters

## Common Parameters
- `data`: Pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
- `tradecost`: Trading cost as percentage (e.g., 0.15 for 0.15%)
- `investment_amount`: Initial investment amount in dollars
- `monthly_investment`: Monthly investment amount for DCA strategy
- `max_investment`: Maximum total investment for DCA strategy

## Main Script (daytrader.py)
The main script orchestrates the analysis by:
1. Loading or fetching stock data
2. Running different trading strategies
3. Analyzing strategy performance
4. Finding optimal parameters
5. Displaying results 