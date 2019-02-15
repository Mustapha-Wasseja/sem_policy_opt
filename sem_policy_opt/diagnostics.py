from collections import OrderedDict
from IPython.display import display, Markdown
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import r2_score

from sem_policy_opt.keras_models import WrappedKerasModel
from sem_policy_opt.market import Market
from sem_policy_opt.run_env import run_env


def pricing_fn_creator(intercept, demand_signal_mult, days_before_flight_mult,
                       seats_avail_mult, competitor_full_mult, price_floor=0):
    '''
    Returns pricing function that is linear combination of function arguments

    Each argument is a list or numpy array of candidate values in output grid
    '''
    def output_fn(demand_signal, days_before_flight, 
                  my_seats_avail, competitor_full):
        base_price = intercept + demand_signal_mult * demand_signal \
                               + days_before_flight_mult * days_before_flight \
                               + seats_avail_mult * my_seats_avail \
                              + competitor_full_mult * competitor_full
        chosen_price = max(base_price, price_floor)
        return chosen_price
    return output_fn

def plot_optim_results(optim_results, 
                       baseline_real_profits=None,
                       baseline_sim_profits=None):
    '''
    Plots scatter plot of predicted and simulated profits. 
    
    Shows plot and returns nothing

    Arguments:
    ----------
    optim_results:  DataFrame with sim_profit and real_profit columns. 
                    A row corresponds to all runs of a single pricing functions
    baseline_real_profits: Real profits from policy used in training data
    baseline_sim_profits: Simulated profits from policy used in training data
    '''
    optim_results.plot.scatter(x='sim_profit', y='real_profit', s=4)
    max_sim_profit = max(optim_results.sim_profit)
    best_result = optim_results.query('sim_profit == @max_sim_profit')

    best_simulated_profit = int(best_result.sim_profit.iloc[0])
    real_attained_profit = int(best_result.real_profit.iloc[0])

    plt.annotate('Profit From Optimized Pricing: ${}'.format(real_attained_profit),
                 xy=(best_simulated_profit, real_attained_profit),
                 xytext=(best_simulated_profit-1, real_attained_profit-15000),
                 arrowprops=dict(width=1, color='r'))

    if baseline_real_profits and baseline_sim_profits:
        plt.annotate('Profit From Strategy Used in Training Data: ${}'.format(int(baseline_real_profits)),
                    xy=(baseline_sim_profits, baseline_real_profits),
                    xytext=(baseline_sim_profits, baseline_real_profits-15000),
                    arrowprops=dict(width=1, color='r'))
    plt.title("Predicted vs Real Profits for Strategies Tried During Optimization")
    plt.xlabel('Profit in Simulation')
    plt.ylabel('Profit in Real Env')
    plt.show()


def r_squared(model, val_data):
    '''
    returns dictionary with r_squared values of model 
    on delta_price, jb_qty_sold and delta_qty_sold

    Arguments:
    ----------
    model: A model accepting input in the format returned by prep_for_pred
    val_data: DataFrame of raw validation data created by run_env
    '''
    preds = model.predict(val_data)
    out = {targ: round(r2_score(val_data[targ].values, pred), 2) 
                    for targ, pred in preds.items()}
    return out

def get_real_and_sim_rewards(real_market, sim_market, pricing_fns, runs_per_fn=15):
    sim_rewards = [run_env(sim_market, pricing_fn, n_times=runs_per_fn)[0].mean()
                        for pricing_fn in pricing_fns]
    real_rewards = [run_env(real_market, pricing_fn, n_times=runs_per_fn)[0].mean()
                        for pricing_fn in pricing_fns]

    results = pd.DataFrame({'sim_profit': sim_rewards,
                            'real_profit': real_rewards})
    return results


def sensitivity_analysis(real_dgp,
                         model_class,
                         market_details,
                         flights_in_training_data,
                         noise_levels,
                         candidate_pricing_fns,
                         baseline_price_fn):
    '''
    Recreates most analysis in original notebook for varying levels of noise
    in the demand signals.

    For each level of noise in noise_levels:
    1) Create a real environment with that amount of demand signal noise, and 
    collect training data from that env
    2) Train predictive model on this training data. Make a sim_env from this
    3) Try all pricing policies in pricing_fns in this sim_env and the real_env 
    created in step 1.  Record resulting profit, and return DF of profits

    Arguments
    ---------
    TODO: write this

    Returns
    -------
    TODO: write this    
    '''
    results = []
    for noise_level in noise_levels:

        alternative_market_details = market_details.copy()
        alternative_market_details['demand_signal_noisiness'] = noise_level
        real_market = Market(real_dgp, alternative_market_details)
        train_profits, train_data = run_env(real_market, 
                                            baseline_price_fn, 
                                            n_times=flights_in_training_data)
        val_profits, val_data = run_env(real_market, 
                                        baseline_price_fn, 
                                        n_times=flights_in_training_data)
        predictive_model = model_class(train_data)
        sim_market = Market(predictive_model, alternative_market_details)

        real_and_sim = get_real_and_sim_rewards(real_market, sim_market, candidate_pricing_fns)
        best_sim_profits = real_and_sim.sim_profit.max()
        real_profits = real_and_sim.real_profit[real_and_sim.sim_profit.idxmax()]
        best_possible_real_profits = real_and_sim.real_profit.max()
        r_squared_vals = r_squared(predictive_model, val_data)

        display(Markdown('---'))
        print("noise level: {}".format(noise_level))
        print("r_squared values: {}".format(r_squared_vals))
        plot_optim_results(real_and_sim)

        results.append(OrderedDict(noise_level = noise_level,
                                   own_qty_r_squared =  r_squared_vals['jb_qty_sold'],
                                   competitor_qty_r_squared = r_squared_vals['delta_qty_sold'],
                                   baseline_profits = int(train_profits.mean()),
                                   real_profits = int(real_profits),
                                   best_sim_profits =  int(best_sim_profits),
                                   best_possible_real_profits = int(best_possible_real_profits)))
    results_df = pd.DataFrame(results)
    return results_df
