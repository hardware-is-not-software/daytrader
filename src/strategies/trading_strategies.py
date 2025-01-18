import pandas as pd

def run_monthly_dca_strategy(data, monthly_investment=1000, max_investment=10000, tradecost=0.15):
    df = data.copy()
    cash = max_investment
    total_invested = 0
    shares = 0
    portfolio_values = []
    trades = []
    
    current_month = None
    
    for index, row in df.iterrows():
        current_price = row['Close']
        current_value = (shares * current_price)
        
        # Record daily portfolio value
        portfolio_values.append({
            'date': index,
            'value': current_value
        })
        
        # Check if this is the first trading day of a new month
        index = pd.to_datetime(index)
        if current_month != index.month:
            current_month = index.month
            
            # Only invest if we haven't reached max investment
            if total_invested < max_investment:
                investment = min(monthly_investment, cash)
                trading_cost = investment * (tradecost / 100)
                shares_to_buy = (investment - trading_cost) / current_price
                shares += shares_to_buy
                cash -= investment
                total_invested += investment
                current_value = shares * current_price
                
                trades.append({
                    'date': index,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'invested': investment,
                    'trading_cost': trading_cost,
                    'total_invested': total_invested,
                    'portfolio_value': current_value
                })
                print(f"DCA BUY: {index.date()}, Price: ${current_price:.2f}, Shares: {shares_to_buy:.2f}, Cost: ${trading_cost:.2f}")
    
    final_value = shares * df['Close'].iloc[-1]
    return pd.DataFrame(trades), final_value, pd.DataFrame(portfolio_values)

def run_dip_recovery_strategy(data, investment_amount=10000, tradecost=0.15):
    df = data.copy()
    df['daily_open'] = df['Open']
    df['intraday_return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    
    cash = investment_amount
    shares = 0
    trades = []
    portfolio_values = []
    entry_price = None
    
    for index, row in df.iterrows():
        current_price = row['Close']
        current_value = cash + (shares * current_price)
        
        portfolio_values.append({
            'date': index,
            'value': current_value
        })
        
        # Buy condition: 2% drop from daily open
        if shares == 0 and row['intraday_return'] <= -2:
            trading_cost = investment_amount * (tradecost / 100)
            shares_to_buy = (investment_amount - trading_cost) / current_price
            if shares_to_buy > 0:
                shares = shares_to_buy
                cash -= (shares * current_price + trading_cost)
                entry_price = row['Open']
                trades.append({
                    'date': index,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': cash + shares * current_price,
                    'trading_cost': trading_cost,
                    'return': row['intraday_return']
                })
                print(f"BUY: {index.date()}, Price: ${current_price:.2f}, Drop: {row['intraday_return']:.1f}%, Cost: ${trading_cost:.2f}")
        
        # Sell condition: price recovered to or above entry price
        elif shares > 0 and current_price >= entry_price:
            sell_value = shares * current_price
            trading_cost = sell_value * (tradecost / 100)
            profit = (current_price - entry_price) * shares - trading_cost
            cash += (sell_value - trading_cost)
            trades.append({
                'date': index,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': cash,
                'trading_cost': trading_cost,
                'profit': profit
            })
            print(f"SELL: {index.date()}, Price: ${current_price:.2f}, Profit: ${profit:.2f}, Cost: ${trading_cost:.2f}")
            shares = 0
            entry_price = None
    
    final_value = cash + (shares * df['Close'].iloc[-1])
    return pd.DataFrame(trades), final_value, pd.DataFrame(portfolio_values)

def run_buy_and_hold_strategy(data, investment_amount=10000, tradecost=0.15):
    df = data.copy()
    initial_price = df['Close'].iloc[0]
    trading_cost = investment_amount * (tradecost / 100)
    shares = (investment_amount - trading_cost) / initial_price
    portfolio_values = []
    
    # Single trade at the beginning
    trades = [{
        'date': df.index[0],
        'action': 'BUY',
        'price': initial_price,
        'shares': shares,
        'value': investment_amount,
        'trading_cost': trading_cost,
        'portfolio_value': investment_amount
    }]
    
    # Calculate daily portfolio values
    for index, row in df.iterrows():
        current_value = shares * row['Close']
        portfolio_values.append({
            'date': index,
            'value': current_value
        })
    
    final_value = shares * df['Close'].iloc[-1]
    print(f"Buy & Hold: {df.index[0].date()}, Price: ${initial_price:.2f}, Shares: {shares:.2f}")
    
    return pd.DataFrame(trades), final_value, pd.DataFrame(portfolio_values)
