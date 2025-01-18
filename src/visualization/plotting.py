import matplotlib.pyplot as plt
import numpy as np
import os

def create_3d_strategy_plot(results, buyhold_final, dca_final, stock, stoploss, tradecost, company_name):
    """Create 3D visualization of strategy performance"""
    # Create stock-specific results directory if it doesn't exist
    os.makedirs(f'results/{stock}', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')
    
    # Find best and worst values
    best_value = np.max(results[:, 2])
    worst_value = np.min(results[:, 2])
    
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
    
    # Add text annotations
    max_y = max(y_range)
    min_x = min(x_range)
    text_y_positions = np.linspace(max_y, max_y - 0.5, 4)
    
    ax.text(min_x, text_y_positions[0], buyhold_final * 1.02, 
            f'Buy & Hold\n${buyhold_final:,.2f}', 
            color='red', fontsize=10, fontweight='bold')
    ax.text(min_x, text_y_positions[1], dca_final * 1.02, 
            f'DCA\n${dca_final:,.2f}', 
            color='green', fontsize=10, fontweight='bold')
    ax.text(min_x, text_y_positions[2], best_value * 1.02, 
            f'Best Dip\n${best_value:,.2f}', 
            color='purple', fontsize=10, fontweight='bold')
    ax.text(min_x, text_y_positions[3], worst_value * 1.02, 
            f'Worst Dip\n${worst_value:,.2f}', 
            color='orange', fontsize=10, fontweight='bold')
    
    # Add horizontal grid lines at reference values
    for ref_value in [buyhold_final, dca_final, best_value, worst_value]:
        ax.plot([min(x_range), max(x_range)], 
                [min(y_range), max(y_range)], 
                [ref_value, ref_value], 
                'k--', alpha=0.3, linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Buy Trigger (%)', labelpad=10)
    ax.set_ylabel('Sell Trigger (%)', labelpad=10)
    ax.set_zlabel('Final Value ($)', labelpad=10)
    ax.set_title(f'\nStrategy Performance Comparison - {company_name} ({stock})\n({abs(stoploss)}% Stop Loss, {tradecost}% Trading Cost)', 
                pad=40, size=14)
    
    # Rotate the view for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add legend
    ax.legend(title='Drop Window Duration')
    
    plt.tight_layout()
    plt.savefig(f'results/{stock}/strategy_3d_{stock}.png')
    plt.close()

def create_strategy_comparison_plot(optimized_values, worst_values, dca_values, hold_values, 
                                  optimized_trades, dca_trades, stock, stoploss, tradecost,
                                  best_days, best_buy, best_sell, company_name):
    """Create time-domain comparison plot of different strategies"""
    # Create stock-specific results directory if it doesn't exist
    os.makedirs(f'results/{stock}', exist_ok=True)
    
    plt.figure(figsize=(15, 7))
    
    # Plot strategy values over time
    plt.plot(optimized_values['date'], optimized_values['value'], 
            label=f'Optimized {best_days}-Day Dip Strategy ({best_buy:.1f}%/{best_sell:.1f}%) - {tradecost}% per trade', 
            color='purple')
    plt.plot(worst_values['date'], worst_values['value'], 
            label=f'Worst Strategy', 
            color='orange', linestyle='--')
    plt.plot(dca_values['date'], dca_values['value'], 
            label=f'Monthly DCA - {tradecost}% per trade', 
            color='green')
    plt.plot(hold_values['date'], hold_values['value'], 
            label=f'Buy & Hold - {tradecost}% once', 
            color='red')
    
    # Add transaction markers
    if len(optimized_trades) > 0:
        opt_buy_trades = optimized_trades[optimized_trades['action'] == 'BUY']
        opt_sell_trades = optimized_trades[optimized_trades['action'] == 'SELL']
        
        if not opt_buy_trades.empty:
            plt.scatter(opt_buy_trades['date'], opt_buy_trades['value'], 
                       color='purple', marker='^', s=100, label='Buy', zorder=5)
        if not opt_sell_trades.empty:
            plt.scatter(opt_sell_trades['date'], opt_sell_trades['value'], 
                       color='magenta', marker='v', s=100, label='Sell', zorder=5)
    
    # Add DCA transaction markers
    if len(dca_trades) > 0:
        plt.scatter(dca_trades['date'], dca_trades['portfolio_value'], 
                   color='green', marker='*', s=100, label='DCA Buy', zorder=5)
    
    plt.title(f'{company_name} ({stock}) Strategy Comparison\nStop Loss: {stoploss}% | Trading Cost: {tradecost}%')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'results/{stock}/strategy_timedomain_{stock}.png')
    plt.close()
