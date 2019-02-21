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

def plot_optim_results(sim_revs, 
                       real_rev,
                       baseline_real_rev=None,
                       extra_title_text=''):
    '''
    Plots scatter plot of predicted and simulated revenue. 
    
    Shows plot and returns nothing

    Arguments:
    ----------
    sim_revs:  Dictionary keyed by simulator name. Values are arrays of revenues
    real_revs: Array of revenue from running policies in real DGP. 
               Order matches sim_revs of results matches ordering of sim_revs
               in terms of the policies that generate each outcome
    baseline_real_rev: Real revenue from policy used in training data
    '''

    assert type(sim_revs) == dict
    num_sims = len(sim_revs)
    fig, ax = plt.subplots(1, num_sims, sharey=True)
    fig.subplots_adjust(top=.8, bottom=0)
    for i, (sim_name, rev) in enumerate(sim_revs.items()):
        if len(sim_revs) > 1:
            my_ax = ax[i]
        else:
            my_ax = ax
        my_ax.scatter(x=rev, y=real_rev, s=4)
        max_sim_rev = int(max(rev))
        real_attained_rev = int(real_rev[rev.argmax()])

        my_ax.annotate('Rev After\nOptimization\n${}'.format(real_attained_rev),
                     fontsize=9, ha='left', va='center',
                     xy=(max_sim_rev, real_attained_rev),
                     xytext=(max_sim_rev-5000, real_attained_rev-15000),
                     arrowprops=dict(width=.5, color='r'))
        my_ax.set_title(sim_name, fontsize=11)
        my_ax.axhline(baseline_real_rev, linestyle=':', linewidth=2)
        fig.suptitle("Predicted vs Real Revenue for Alternative Pricing Strategies" + \
                     "\n" + extra_title_text)
        fig.text(0.5, -.1, 'Revenue in Simulator', ha='center')
        fig.text(-0.1, 0.5, 'Revenue in Real Env', va='center', rotation='vertical')
    my_ax.annotate('Baseline Policy Rev\n${}'.format(int(baseline_real_rev)),
                    fontsize=9, 
                    xy=(max_sim_rev+1000, baseline_real_rev),
                    va='center')
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

def eval_pricing_fns(market, pricing_fns, runs_per_agent=20):
    rewards = [run_env(market, pricing_fn, n_times=runs_per_agent)[0].mean()
                for pricing_fn in pricing_fns]
    return np.array(rewards)


def comparative_analysis(real_dgp,
                         models,
                         market_details,
                         flights_in_training_data,
                         noise_levels,
                         pricing_fns,
                         baseline_price_fn):
    '''
    Creates simulators for each noise_level x model class pair. Tests each pricing_fn
    Displays scatter depicting results

    Arguments
    ---------
    TODO: write this

    Returns
    -------
    TODO: write this    
    '''
    results = []
    for noise_level in noise_levels:

        display(Markdown('---'))
        print("noise level: {}".format(noise_level))

        alternative_market_details = market_details.copy()
        alternative_market_details['demand_signal_noisiness'] = noise_level
        real_market = Market(real_dgp, alternative_market_details)
        # results for each pricing function in real_market
        real_results = eval_pricing_fns(real_market, pricing_fns)
        train_rev, train_data = run_env(real_market, 
                                            baseline_price_fn, 
                                            n_times=flights_in_training_data)
        all_results_by_model = {}
        real_rev_by_model = {}
        top_pred_rev_by_model = {}
        for model_name, model_class in models.items():
            predictive_model = model_class(train_data)

            # val_data is used only to calculate r^2
            _, val_data = run_env(real_market, 
                                  baseline_price_fn, 
                                  n_times=flights_in_training_data)
            r_squared_vals = r_squared(predictive_model, val_data)
            print("R-squared values in {} model: {}".format(model_name, r_squared_vals))

            sim_market = Market(predictive_model, alternative_market_details)
            sim_results = eval_pricing_fns(sim_market, pricing_fns)
            all_results_by_model[model_name] = sim_results
            top_pred_rev_by_model[model_name] = sim_results.max()
            real_rev_by_model[model_name] = real_results[sim_results.argmax()]
        
        plot_optim_results(all_results_by_model, 
                          real_results, 
                          train_rev.mean(),
                          "Noise Level: {}".format(noise_level))
        summary_this_noise = {'noise_level': noise_level,
                              'best_possible_real_rev': int(real_results.max())
                             }
        for model_name in models.keys():
            summary_this_noise[model_name + '_pred_rev'] = int(top_pred_rev_by_model[model_name]),
            summary_this_noise[model_name + '_optimized_rev'] = int(real_rev_by_model[model_name])
        results.append(summary_this_noise)
    overview_df = pd.DataFrame(results).set_index('noise_level')
    return overview_df

def model_comparison_plot(comparative_results):
    comparative_results.Bayesian_optimized_rev.plot()
    comparative_results.Conventional_optimized_rev.plot()
    plt.legend(['Bayesian', 'Conventional'])
    plt.title('Optimized Policy Revenue at Each Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Revenue in DGP')
    plt.show()