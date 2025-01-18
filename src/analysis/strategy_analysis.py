import numpy as np
import pandas as pd

def calculate_max_drawdown(values):
    peak = values.expanding(min_periods=1).max()
    drawdown = ((values - peak) / peak) * 100
    return drawdown.min()

def run_strategy_with_parameters(data, buy_trigger, sell_trigger, days_window=1, investment_amount=10000, trading_cost=15):
    df = data.copy()
    df['daily_open'] = df['Open']
    
    # Calculate rolling price change over specified days window
    df['price_change'] = ((df['Close'] - df['Close'].shift(days_window)) / df['Close'].shift(days_window) * 100)
    
    cash = investment_amount
    shares = 0
    entry_price = None
    
    for index, row in df.iterrows():
        current_price = row['Close']
        
        # Buy condition: specified drop over the last n days
        if shares == 0 and not pd.isna(row['price_change']) and row['price_change'] <= buy_trigger:
            shares_to_buy = (investment_amount - trading_cost) / current_price
            if shares_to_buy > 0:
                shares = shares_to_buy
                cash -= (shares * current_price + trading_cost)
                entry_price = current_price
        
        # Sell conditions: target reached or stop loss hit
        elif shares > 0:
            current_return = ((current_price - entry_price) / entry_price) * 100
            if current_return >= sell_trigger:
                cash += (shares * current_price - trading_cost)
                shares = 0
                entry_price = None
    
    final_value = cash + (shares * df['Close'].iloc[-1])
    return final_value

def create_3d_analysis(data, trigger_resolution=0.5, max_buytrigger=5, max_selltrigger=5):
    # Generate trigger ranges
    buy_triggers = np.arange(-trigger_resolution, -max_buytrigger - trigger_resolution, -trigger_resolution)
    sell_triggers = np.arange(trigger_resolution, max_selltrigger + trigger_resolution, trigger_resolution)
    days_windows = [1, 2, 3, 4, 5]
    
    results = []
    for days in days_windows:
        for buy in buy_triggers:
            for sell in sell_triggers:
                final_value = run_strategy_with_parameters(data, buy, sell, days)
                results.append([buy, sell, final_value, days])
    
    return np.array(results) 