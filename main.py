import altair as alt
from matplotlib import pyplot as plt
import pandas as pd
from theano import shared

from market import RealMarket, SimulatedMarket
from keras_models import get_keras_model, prep_data_for_keras_model
from pymc_models import WrappedPymcModel
from model_testing import test_jb_price_rule



plt.style.use('fivethirtyeight')


max_demand_level = 500
demand_signal_noisiness = 10
customer_level_randomness = 50
seats_per_flight = 200
sales_window_length = 100

def naive_price_fn(my_demand_signal, opponent_demand_signal, days_before_flight, my_seats_avail, competitor_seats_avail): 
    # Charge more if you have a lot of time to sell seats, few seats are available, or you have little competition
    # On net, prices may increase over time because low seat inventory overwhelms remaining time effect.
    return max(my_demand_signal, days_before_flight) + 2 * days_before_flight - my_seats_avail - 25 * (competitor_seats_avail==0)


market = RealMarket(max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length, 
                    jetblue_price_fn=naive_price_fn, delta_price_fn=naive_price_fn, potential_customers_per_day=20, 
                    customer_level_randomness=customer_level_randomness)
train_data = pd.concat([market.simulate_flight_sales() for _ in range(100)])
val_data = pd.concat([market.simulate_flight_sales() for _ in range(10)])

train_x, train_y = prep_data_for_keras_model(train_data)
val_x, val_y = prep_data_for_keras_model(val_data)
keras_model = get_keras_model(train_x, train_y, val_x, val_y)

print("Finished fitting keras model")

naive_price_rule_test_sim = test_jb_price_rule(naive_price_fn, max_demand_level, demand_signal_noisiness, seats_per_flight, 
                                               sales_window_length=100, n_sims=100, q_and_delta_price_model=keras_model)
print("Finished test in simulator")
naive_price_rule_test_real = test_jb_price_rule(naive_price_fn, max_demand_level, demand_signal_noisiness, seats_per_flight, 
                                                sales_window_length=100, n_sims=100, customer_level_randomness=customer_level_randomness, 
                                                delta_price_fn=naive_price_fn)
print("Finished test in data generating environment")
tests_data = pd.concat([pd.DataFrame({'rev': naive_price_rule_test_sim, 'source': 'simulator'}), 
                        pd.DataFrame({'rev': naive_price_rule_test_real, 'source': 'true_env'})])

true_pred_comp = alt.Chart(tests_data).mark_area(opacity=0.3, interpolate='step').encode(
                alt.X('rev', bin=True),
                alt.Y('count()', stack=None),
                alt.Color('source'))

#true_pred_comp.serve()


run_pymc_model = False
if run_pymc_model:

    pmodel = WrappedPymcModel(train_data)

    ## TODO: Solve inability to get predictions from pymc model because delta_price being observed at training time forces dimensionality
    ##       of delta_price at test time to be of same dimension.  EG, following line doesn't work.
    test_1 = pmodel.predict([50, 50, 10, 50, 50])
    ## Additionally, the values for parameters used in current predict step seem to be the mean values of prior. Possibly getting pulled
    ## from trace?