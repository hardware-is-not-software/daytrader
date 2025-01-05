import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import os
import sys

def save_to_csv(data, filename='stock_data.csv'):
    data.to_csv(filename)

def load_from_csv(filename='stock_data.csv'):
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    return None

def get_csv_filename(ticker):
    """Generate CSV filename for a given ticker"""
    return f"{ticker.upper()}.csv"

def get_daily_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    try:
        ticker = yf.Ticker(args.stock)
        data = ticker.history(start=start_date, end=end_date, interval="1d")
        if len(data) == 0:
            print(f"\nError: No data found for ticker '{args.stock}'")
            print("Please verify the stock symbol is correct (e.g., AAPL, MSFT, GOOGL)")
            sys.exit(1)
        return data
    except Exception as e:
        print(f"\nError: Failed to fetch data for ticker '{args.stock}'")
        print("Please verify the stock symbol is correct (e.g., AAPL, MSFT, GOOGL)")
        print(f"Error details: {str(e)}")
        sys.exit(1)

def validate_data(data):
    """Validate that we have proper stock data"""
    if data is None or len(data) == 0:
        print(f"\nError: No valid data found for ticker '{args.stock}'")
        print("Please delete the CSV file and try again to fetch fresh data.")
        sys.exit(1)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"\nError: Missing required columns in data: {missing_columns}")
        print("Please delete the CSV file and try again to fetch fresh data.")
        sys.exit(1)

def run_monthly_dca_strategy(data, monthly_investment=1000, max_investment=10000):
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
        if current_month != index.month:
            current_month = index.month
            
            # Only invest if we haven't reached max investment
            if total_invested < max_investment:
                investment = min(monthly_investment, cash)
                trading_cost = investment * (args.tradecost / 100)
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

def run_dip_recovery_strategy(data, investment_amount=10000):
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
            trading_cost = investment_amount * (args.tradecost / 100)
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
            trading_cost = sell_value * (args.tradecost / 100)
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

def run_buy_and_hold_strategy(data, investment_amount=10000):
    df = data.copy()
    initial_price = df['Close'].iloc[0]
    trading_cost = investment_amount * (args.tradecost / 100)
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
            if current_return >= sell_trigger or current_return <= args.stoploss:
                cash += (shares * current_price - trading_cost)
                shares = 0
                entry_price = None
    
    final_value = cash + (shares * df['Close'].iloc[-1])
    return final_value

def create_3d_analysis(data):
    # Generate trigger ranges based on command line parameters
    buy_triggers = np.arange(-args.trigger_resolution, -args.max_buytrigger - args.trigger_resolution, -args.trigger_resolution)
    sell_triggers = np.arange(args.trigger_resolution, args.max_selltrigger + args.trigger_resolution, args.trigger_resolution)
    days_windows = [1, 2, 3, 4, 5]
    
    results = []
    for days in days_windows:
        for buy in buy_triggers:
            for sell in sell_triggers:
                final_value = run_strategy_with_parameters(data, buy, sell, days)
                results.append([buy, sell, final_value, days])
    
    return np.array(results)

def find_best_parameters(results):
    """Find the buy/sell triggers that produced the highest final value"""
    best_idx = np.argmax(results[:, 2])
    return results[best_idx]

# Replace the argument parser section with:
parser = argparse.ArgumentParser(
    description='''Stock trading strategy analyzer that compares DCA, Buy & Hold, and Dip-buying strategies.
    
Example usage:
    python daytrader.py --stock INTC --stoploss -15 --tradecost 0.2
    
Parameters:
    --stock     : Stock ticker symbol (e.g., TSLA, AAPL, INTC)
    --stoploss  : Stop loss percentage as negative number (e.g., -15 for 15%)
    --tradecost : Trading cost as percentage (e.g., 0.2 for 0.2%)
''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('--stock', type=str, required=True, 
                   help='Stock ticker symbol (required). Example: INTC')
parser.add_argument('--stoploss', type=float, required=True, 
                   help='Stop loss percentage as negative number (required). Example: -15')
parser.add_argument('--tradecost', type=float, required=True,
                   help='Trading cost as percentage (required). Example: 0.2')
parser.add_argument('--max_buytrigger', type=float, default=5.0,
                   help='Maximum buy trigger percentage (positive). Default: 4.0')
parser.add_argument('--max_selltrigger', type=float, default=5.0,
                   help='Maximum sell trigger percentage (positive). Default: 4.0')
parser.add_argument('--trigger_resolution', type=float, default=0.5,
                   help='Step size between trigger values. Default: 0.5')

# Replace the argument parsing section with try-except block
try:
    args = parser.parse_args()
except SystemExit as e:
    if str(e) != '0':  # When not --help
        print('''\nExample usage:
    python daytrader.py --stock INTC --stoploss -15 --tradecost 0.2
    
Required parameters:
    --stock     : Stock ticker symbol (e.g., TSLA, AAPL, INTC)
    --stoploss  : Stop loss percentage as negative number (e.g., -15 for 15%)
    --tradecost : Trading cost as percentage (e.g., 0.2 for 0.2%)\n''')
    raise

# Enhanced error messages
if args.stoploss >= 0:
    parser.error('''Stop loss must be a negative number.
Example usage:
    python daytrader.py --stock INTC --stoploss -5 --tradecost 0.2''')

if args.tradecost <= 0:
    parser.error('''Trading cost must be a positive number.
Example usage:
    python daytrader.py --stock INTC --stoploss -5 --tradecost 0.2''')

print(f"Analyzing {args.stock} with {abs(args.stoploss)}% stop loss and {args.tradecost}% trading cost...")

# Get data and run strategies
csv_filename = get_csv_filename(args.stock)
if os.path.exists(csv_filename):
    print(f"Loading existing data for {args.stock} from {csv_filename}")
    try:
        data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
        validate_data(data)
    except Exception as e:
        print(f"\nError: Failed to load data from {csv_filename}")
        print(f"Error details: {str(e)}")
        print("Please delete the CSV file and try again to fetch fresh data.")
        sys.exit(1)
else:
    print(f"Fetching daily data for {args.stock}...")
    data = get_daily_data()
    validate_data(data)
    data.to_csv(csv_filename)

print("Running strategies...")

# Replace the 3D plotting section:
if True:
    print("\nRunning 3D Strategy Analysis...")
    results = create_3d_analysis(data)
    
    # Run reference strategies first
    _, buyhold_final, _ = run_buy_and_hold_strategy(data)
    _, dca_final, _ = run_monthly_dca_strategy(data)
    
    # Find best and worst values before plotting
    best_value = np.max(results[:, 2])
    worst_value = np.min(results[:, 2])
    
    # Create single 3D plot with all time windows
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different day windows
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Plot each time window with different color
    for idx, days_window in enumerate([1, 2, 3, 4, 5]):
        window_results = results[results[:, 3] == days_window]
        
        x = window_results[:, 0]  # Buy triggers
        y = window_results[:, 1]  # Sell triggers
        z = window_results[:, 2]  # Final values
        
        scatter = ax.scatter(x, y, z, c=colors[idx], 
                           label=f'{days_window}-Day Window',
                           s=200, alpha=0.6)
        
        # Add lines connecting points to the bottom
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y[i], y[i]], 
                   [min(results[:, 2]), z[i]], 
                   color=colors[idx], alpha=0.1)
    
    # Add reference planes for Buy & Hold and DCA
    x_range = np.array([min(results[:, 0]), max(results[:, 0])])
    y_range = np.array([min(results[:, 1]), max(results[:, 1])])
    X, Y = np.meshgrid(x_range, y_range)
    
    # Buy & Hold reference plane
    Z_buyhold = np.full_like(X, buyhold_final)
    ax.plot_surface(X, Y, Z_buyhold, alpha=0.3, color='red', 
                   label='Buy & Hold', shade=False)
    
    # DCA reference plane
    Z_dca = np.full_like(X, dca_final)
    ax.plot_surface(X, Y, Z_dca, alpha=0.3, color='green', 
                   label='DCA', shade=False)
    
    # Add clear text annotations for all reference values
    text_y_positions = np.linspace(max(y), max(y) - 0.5, 4)  # Space out annotations
    ax.text(min(x), text_y_positions[0], buyhold_final * 1.02, 
            f'Buy & Hold\n${buyhold_final:,.2f}', 
            color='red', fontsize=10, fontweight='bold')
    ax.text(min(x), text_y_positions[1], dca_final * 1.02, 
            f'DCA\n${dca_final:,.2f}', 
            color='green', fontsize=10, fontweight='bold')
    ax.text(min(x), text_y_positions[2], best_value * 1.02, 
            f'Best Dip\n${best_value:,.2f}', 
            color='purple', fontsize=10, fontweight='bold')
    ax.text(min(x), text_y_positions[3], worst_value * 1.02, 
            f'Worst Dip\n${worst_value:,.2f}', 
            color='orange', fontsize=10, fontweight='bold')
    
    # Add horizontal grid lines at reference values
    for ref_value in [buyhold_final, dca_final, best_value, worst_value]:
        ax.plot([min(x), max(x)], [min(y), max(y)], [ref_value, ref_value], 
                'k--', alpha=0.3, linewidth=1)
    
    # Customize the plot with single-line title but add empty line first
    ax.set_xlabel('Buy Trigger (%)', labelpad=10)
    ax.set_ylabel('Sell Trigger (%)', labelpad=10)
    ax.set_zlabel('Final Value ($)', labelpad=10)
    ax.set_title(f'\nStrategy Performance Comparison - {args.stock} ({abs(args.stoploss)}% Stop Loss, {args.tradecost}% Trading Cost)', 
                pad=40, size=14)
    
    # Rotate the view for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add legend
    ax.legend(title='Drop Window Duration')
    
    plt.tight_layout()
    plt.show()

    # Print best results for each window
    print("\nBest Strategy for Each Time Window:")
    print("Days | Buy Trigger | Sell Trigger | Final Value  | vs Buy&Hold | vs DCA")
    print("-" * 80)
    
    for days_window in [1, 2, 3, 4, 5]:
        window_results = results[results[:, 3] == days_window]
        best_idx = np.argmax(window_results[:, 2])
        best = window_results[best_idx]
        
        vs_buyhold = ((best[2] - buyhold_final) / buyhold_final * 100)
        vs_dca = ((best[2] - dca_final) / dca_final * 100)
        print(f"{int(best[3]):4d} | {best[0]:10.1f}% | {best[1]:11.1f}% | "
              f"${best[2]:10,.2f} | {vs_buyhold:9.1f}% | {vs_dca:6.1f}%")

# Find the best overall strategy across all time windows
best_result = None
best_value = 0

for days_window in [1, 2, 3, 4, 5]:
    window_results = results[results[:, 3] == days_window]
    best_idx = np.argmax(window_results[:, 2])
    best = window_results[best_idx]
    
    if best[2] > best_value:
        best_value = best[2]
        best_result = [best[0], best[1], best[2], int(best[3])]  # buy_trigger, sell_trigger, final_value, days

print(f"\nBest Overall Strategy:")
print(f"Time Window: {best_result[3]} days")
print(f"Buy Trigger: {best_result[0]:.1f}%")
print(f"Sell Trigger: {best_result[1]:.1f}%")  # Fixed format specifier
print(f"Final Value: ${best_result[2]:,.2f}")

def run_optimized_dip_strategy(data, buy_trigger, sell_trigger, days_window=1, investment_amount=10000, trading_cost=15):
    """Similar to run_dip_recovery_strategy but with configurable triggers and stop loss"""
    df = data.copy()
    df['daily_open'] = df['Open']
    df['price_change'] = ((df['Close'] - df['Close'].shift(days_window)) / df['Close'].shift(days_window) * 100)
    
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
        
        # Buy condition
        if shares == 0 and not pd.isna(row['price_change']) and row['price_change'] <= buy_trigger:
            trading_cost = investment_amount * (args.tradecost / 100)
            shares_to_buy = (investment_amount - trading_cost) / current_price
            if shares_to_buy > 0:
                shares = shares_to_buy
                cash -= (shares * current_price + trading_cost)
                entry_price = current_price
                trades.append({
                    'date': index,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': cash + shares * current_price,
                    'trading_cost': trading_cost,
                    'return': row['price_change']
                })
                print(f"OPTIMIZED BUY ({days_window}-day drop): {index.date()}, Price: ${current_price:.2f}, Drop: {row['price_change']:.1f}%, Cost: ${trading_cost:.2f}")
        
        # Sell conditions: target reached or stop loss hit
        elif shares > 0:
            current_return = ((current_price - entry_price) / entry_price) * 100
            if current_return >= sell_trigger or current_return <= args.stoploss:
                sell_value = shares * current_price
                trading_cost = sell_value * (args.tradecost / 100)
                profit = (current_price - entry_price) * shares - trading_cost
                cash += (sell_value - trading_cost)
                trades.append({
                    'date': index,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': cash,
                    'trading_cost': trading_cost,
                    'profit': profit,
                    'return': current_return,
                    'stop_loss_triggered': current_return <= args.stoploss
                })
                print(f"OPTIMIZED SELL: {index.date()}, Price: ${current_price:.2f}, Return: {current_return:.1f}%, Profit: ${profit:.2f}, Cost: ${trading_cost:.2f}, Stop Loss: {current_return <= args.stoploss}")
                shares = 0
                entry_price = None
    
    final_value = cash + (shares * df['Close'].iloc[-1])
    return pd.DataFrame(trades), final_value, pd.DataFrame(portfolio_values)

# Run strategies with best parameters
dca_trades, dca_final_value, dca_values = run_monthly_dca_strategy(data)
optimized_trades, optimized_final_value, optimized_values = run_optimized_dip_strategy(
    data, 
    buy_trigger=best_result[0],
    sell_trigger=best_result[1],
    days_window=best_result[3]
)
hold_trades, hold_final_value, hold_values = run_buy_and_hold_strategy(data)

# Before creating fig2, find worst performing strategy
worst_result = None
worst_value = float('inf')

for days_window in [1, 2, 3, 4, 5]:
    window_results = results[results[:, 3] == days_window]
    worst_idx = np.argmin(window_results[:, 2])
    worst = window_results[worst_idx]
    
    if worst[2] < worst_value:
        worst_value = worst[2]
        worst_result = [worst[0], worst[1], worst[2], int(worst[3])]

# Run worst strategy
worst_trades, worst_final_value, worst_values = run_optimized_dip_strategy(
    data, 
    buy_trigger=worst_result[0],
    sell_trigger=worst_result[1],
    days_window=worst_result[3]
)

# Update the strategy comparison plot section
fig2 = plt.figure(figsize=(15, 7))
plt.plot(optimized_values['date'], optimized_values['value'], 
         label=f'Optimized {best_result[3]}-Day Dip Strategy ({best_result[0]:.1f}%/{best_result[1]:.1f}%) - {args.tradecost}% per trade', 
         color='purple')
plt.plot(worst_values['date'], worst_values['value'], 
         label=f'Worst {worst_result[3]}-Day Strategy ({worst_result[0]:.1f}%/{worst_result[1]:.1f}%)', 
         color='orange', linestyle='--')
plt.plot(dca_values['date'], dca_values['value'], 
         label=f'Monthly DCA - {args.tradecost}% per trade', 
         color='green')
plt.plot(hold_values['date'], hold_values['value'], 
         label=f'Buy & Hold - {args.tradecost}% once', 
         color='red')

# Add optimized strategy transaction markers
if len(optimized_trades) > 0:
    opt_buy_trades = optimized_trades[optimized_trades['action'] == 'BUY']
    opt_sell_trades = optimized_trades[optimized_trades['action'] == 'SELL']
    
    if not opt_buy_trades.empty:
        plt.scatter(opt_buy_trades['date'], opt_buy_trades['value'], 
                   color='purple', marker='^', s=100, label='Buy', zorder=5)
    if not opt_sell_trades.empty:
        plt.scatter(opt_sell_trades['date'], opt_sell_trades['value'], 
                   color='magenta', marker='v', s=100, label='Sell', zorder=5)
        
        # Mark stop loss events
        stop_loss_trades = opt_sell_trades[opt_sell_trades['stop_loss_triggered'] == True]
        if not stop_loss_trades.empty:
            plt.scatter(stop_loss_trades['date'], stop_loss_trades['value'],
                       color='black', marker='x', s=100, label='Stop Loss', zorder=6)

# Add DCA transaction markers
if len(dca_trades) > 0:
    plt.scatter(dca_trades['date'], dca_trades['portfolio_value'], 
               color='green', marker='*', s=100, label='DCA Buy', zorder=5)

# Update title to include stock symbol, stop loss and trading cost
plt.title(f'{args.stock} Strategy Comparison\nStop Loss: {args.stoploss}% | Trading Cost: {args.tradecost}%')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print results
print("\nResults Comparison:")
print("==================")

print(f"\nOptimized {best_result[3]}-Day Dip Strategy (Buy: {best_result[0]:.1f}%, Sell: {best_result[1]:.1f}%):")
print(f"Final value: ${optimized_final_value:.2f}")
print(f"Total return: {((optimized_final_value - 10000) / 10000 * 100):.2f}%")
if len(optimized_trades) > 0:
    print(f"Number of trades: {len(optimized_trades)}")
    print(f"Total trading costs: ${len(optimized_trades) * 15}")
    
    # Calculate win rate for completed trades
    profitable_trades = optimized_trades[optimized_trades['action'] == 'SELL']
    if len(profitable_trades) > 0:
        win_rate = (profitable_trades['profit'] > 0).mean() * 100
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Average profit per trade: ${profitable_trades['profit'].mean():.2f}")

print(f"\nWorst {worst_result[3]}-Day Dip Strategy (Buy: {worst_result[0]:.1f}%, Sell: {worst_result[1]:.1f}%):")
print(f"Final value: ${worst_final_value:.2f}")
print(f"Total return: {((worst_final_value - 10000) / 10000 * 100):.2f}%")
if len(worst_trades) > 0:
    print(f"Number of trades: {len(worst_trades)}")
    print(f"Total trading costs: ${worst_trades['trading_cost'].sum():.2f}")

print(f"\nMonthly DCA Strategy:")
print(f"Final value: ${dca_final_value:.2f}")
print(f"Total return: {((dca_final_value - 10000) / 10000 * 100):.2f}%")
print(f"Number of investments: {len(dca_trades)}")
print(f"Total trading costs: ${len(dca_trades) * 15}")

print(f"\nBuy & Hold Strategy:")
print(f"Final value: ${hold_final_value:.2f}")
print(f"Total return: {((hold_final_value - 10000) / 10000 * 100):.2f}%")
print(f"Number of trades: 1")
print(f"Total trading costs: $15")

# Update drawdown calculation
print(f"\nMaximum Drawdown:")
print(f"Optimized Dip Strategy: {calculate_max_drawdown(optimized_values['value']):.2f}%")
print(f"Monthly DCA: {calculate_max_drawdown(dca_values['value']):.2f}%")
print(f"Buy & Hold: {calculate_max_drawdown(hold_values['value']):.2f}%")  # Fixed format specifier

# Add cumulative trading costs summary
total_opt_cost = optimized_trades['trading_cost'].sum() if len(optimized_trades) > 0 else 0
total_dca_cost = dca_trades['trading_cost'].sum() if len(dca_trades) > 0 else 0
total_hold_cost = hold_trades['trading_cost'].sum() if len(hold_trades) > 0 else 0

print(f"\nCumulative Trading Costs:")
print(f"Optimized Strategy: ${total_opt_cost:.2f}")
print(f"Monthly DCA: ${total_dca_cost:.2f}")
print(f"Buy & Hold: ${total_hold_cost:.2f}")
