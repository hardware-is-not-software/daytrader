from src.utils.data_handlers import store_facts_to_file
from src.analysis.strategy_analysis import calculate_max_drawdown
def store_analysis_results(args, dca_final, dca_values, dip_final, dip_values, bh_final, bh_values, 
                       best_value, best_buy, best_sell, best_days, 
                       worst_value, worst_buy, worst_sell, worst_days,
                       optimized_trades, dip_trades, bh_trades, dca_trades, 
                       stock, company_name):
    facts = {
        "parameters": vars(args),
        "strategy_results": {
            "dca": {
                "final_value": dca_final,
                "max_drawdown": calculate_max_drawdown(dca_values['value']),
                "trades": len(dca_trades),
                "parameters": {
                    "frequency": "monthly",
                    "monthly_investment": 1000,
                    "max_investment": 10000
                }
            },
            "buy_and_hold": {
                "final_value": bh_final,
                "max_drawdown": calculate_max_drawdown(bh_values['value']),
                "trades": len(bh_trades),
                "parameters": {
                    "investment_amount": 10000
                }
            },
            "dip_strategy_best": {
                "final_value": best_value,
                "max_drawdown": calculate_max_drawdown(dip_values['value']),
                "trades": len(optimized_trades),
                "parameters": {
                    "buy_trigger": best_buy,
                    "sell_trigger": best_sell,
                    "days_window": int(best_days),
                    "investment_amount": 10000
                }
            },
            "dip_strategy_worst": {
                "final_value": worst_value,
                "max_drawdown": calculate_max_drawdown(dip_values['value']),
                "trades": len(dip_trades),
                "parameters": {
                    "buy_trigger": worst_buy,
                    "sell_trigger": worst_sell,
                    "days_window": int(worst_days),
                    "investment_amount": 10000
                }
            }
        },
        "company_name": company_name
    }
    store_facts_to_file(facts, stock)
