import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from keras_models import prep_for_keras_model

def run_env(testing_market, price_fn, n_times=1):

    all_data = []
    all_rewards = []

    for _ in range(n_times):
        obs = testing_market.reset()
        episode_over = False
        cum_reward = 0

        while not episode_over:
            demand_signal, days_before_flight, my_seats_avail, competitor_full = obs
            my_price = price_fn(demand_signal, days_before_flight, my_seats_avail, competitor_full)
            obs, reward, episode_over, debug_info = testing_market.step(my_price)
            cum_reward += reward
        all_rewards.append(cum_reward)
        all_data.append(testing_market.data_df)
    return pd.Series(all_rewards), pd.concat(all_data)

def test_pricing_multipliers(baseline_price_fn, multipliers, sim_market, real_market):

    def pricing_fn_maker(multiplier):
        def new_price_fn(*args, **kwargs):
            return multiplier * baseline_price_fn(*args, **kwargs)
        return new_price_fn

    alternative_pricing_profits = []
    alternative_pricing_scenario_details = {}

    for mult in multipliers:
        new_price_fn = pricing_fn_maker(mult)
        pred_profits, pred_data = run_env(sim_market, new_price_fn, n_times=20)
        actual_profits, actual_data = run_env(real_market, new_price_fn, n_times=20)
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
    