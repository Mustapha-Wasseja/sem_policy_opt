from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from sem_policy_opt.keras_models import prep_for_keras_model
from sem_policy_opt.run_env import run_env


def pricing_fn_creator(intercept, demand_signal_mult, days_before_flight_mult, seats_avail_mult, competitor_full_mult, price_floor=0):
    def output_fn(demand_signal, days_before_flight, my_seats_avail, competitor_full):
        base_price = intercept + demand_signal_mult * demand_signal \
                               + days_before_flight_mult * days_before_flight \
                               + seats_avail_mult * my_seats_avail \
                              + competitor_full_mult * competitor_full
        chosen_price = max(base_price, price_floor)
        return chosen_price
    return output_fn

def plot_optim_results(optim_results, baseline_real_profits, baseline_sim_profits):
    optim_results.plot.scatter(x='sim_profit', y='real_profit')
    max_sim_profit = max(optim_results.sim_profit)
    best_result = optim_results.query('sim_profit == @max_sim_profit')

    best_simulated_profit = int(best_result.sim_profit.iloc[0]) 
    real_attained_profit = int(best_result.real_profit.iloc[0])

    plt.annotate('Profit From Optimized Pricing: ${}'.format(real_attained_profit),
                 xy=(best_simulated_profit, real_attained_profit),
                 xytext=(65000, 60000),
                 arrowprops=dict(width=3, color='r'))

    plt.annotate('Profit From Strategy in Used Training Data: ${}'.format(int(baseline_real_profits)),
                 xy=(baseline_sim_profits, baseline_real_profits),
                 xytext=(30000, 20000),
                 arrowprops=dict(width=2, color='r'))
    plt.title("Predicted vs Real Profits for Strategies Tried During Optimization")
    plt.show()
def test_pricing_multipliers(baseline_price_fn, multipliers, sim_market, real_market, n_sims=20):

    def pricing_fn_maker(multiplier):
        def new_price_fn(*args, **kwargs):
            return multiplier * baseline_price_fn(*args, **kwargs)
        return new_price_fn

    alternative_pricing_profits = []
    alternative_pricing_scenario_details = {}

    for mult in multipliers:
        new_price_fn = pricing_fn_maker(mult)
        pred_profits, pred_data = run_env(sim_market, new_price_fn, n_times=n_sims)
        actual_profits, actual_data = run_env(real_market, new_price_fn, n_times=n_sims)
        alternative_pricing_profits.append((mult, pred_profits.mean(), actual_profits.mean()))
        alternative_pricing_scenario_details[mult] = {'pred': pred_data,
                                                      'actual': actual_data}

    price_comparison = pd.DataFrame(alternative_pricing_profits, 
                                    columns=['base_price_mult', 'mean_predicted_rev', 'mean_actual_rev'])

    return price_comparison

def r_squared(model, val_data):
    val_x = prep_for_keras_model(val_data, skip_y=True)
    preds = model.predict(val_x)
    return {'delta_price_r2': r2_score(val_data.delta_price.values, preds[0].ravel()),
            'jb_qty_sold_r2': r2_score(val_data.jetblue_seats_sold.values, preds[1].ravel()),
            'delta_qty_sold_r2': r2_score(val_data.delta_seats_sold.values, preds[2].ravel())}
    