from matplotlib import pyplot as plt
import numpy as np

from market import RealMarket, SimulatedMarket

def test_jb_price_rule(jetblue_price_fn, max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length=80, n_sims=5, customer_level_randomness=None, delta_price_fn=None, q_and_delta_price_model=None):
    if q_and_delta_price_model is None:     # Run a "real" environment
        sample_environments = [RealMarket(max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length, jetblue_price_fn, delta_price_fn, potential_customers_per_day=20, customer_level_randomness=customer_level_randomness) for _ in range(n_sims)]
    else:                                   # Run a simulated environment
        sample_environments = [SimulatedMarket(max_demand_level, demand_signal_noisiness, 
                                               seats_per_flight, sales_window_length, 
                                               jetblue_price_fn, q_and_delta_price_model) for _ in range(n_sims)]
    profits_in_sims = np.array([env.simulate_flight_sales().jetblue_revenue.sum() for env in sample_environments])
    return profits_in_sims

def plot_profit_dist(x):
    plt.clf()
    plt.hist(x)
    plt.xlabel('Profit')
    plt.title(r'Histogram Of Profits In Simulations')
    plt.axvline(x.mean(), color='r', linewidth=2)
    _, ymax = plt.ylim()
    plt.text(x.mean() + x.mean()/20, 
             ymax - ymax/5, 
             'Mean: {:.2f}'.format(x.mean()))
    plt.show()


# CONTENTS BELOW ARE OUTDATED
def delta_price_pred_fn_from_coeffs(production_intercept, cross_signal_multiplier, days_before_multiplier, sd):
    def choice_fn(my_demand_signal, cross_demand_signal, days_before_flight, my_seats_avail, competitor_flight_full): 
        production = production_intercept + cross_demand_signal * cross_signal_multiplier + days_before_flight * days_before_multiplier+ np.random.randn() * sd ** .5
        clipped_production = np.clip(production, 0, max_demand_level)
        return clipped_production
    return choice_fn

#sample_delta_price_fns_pymc_model = [delta_price_pred_fn_from_coeffs(c.delta_production_intercept, 
#                                                    c.demand_signal_multiplier_cross,
#                                                    c.days_before_multiplier,
#                                                     c.sd_delta) 
#                          for c in posterior_samples.itertuples()]

