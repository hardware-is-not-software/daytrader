from src.utils.data_handlers import store_facts_to_file
from src.analysis.strategy_analysis import calculate_max_drawdown
from src.utils.ticker_utils import get_company_name

def store_analysis_results(args, dca_final, dca_values, dip_final, dip_values, bh_final, bh_values, best_value, best_buy, best_sell, best_days, optimized_trades, dip_trades, bh_trades, dca_trades, stock):
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
        },
        "company_name": get_company_name(stock)
    }
    store_facts_to_file(facts, stock)
