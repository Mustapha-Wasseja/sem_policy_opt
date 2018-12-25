import altair as alt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('fivethirtyeight')


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

def plot_profit_comparison(simulator_profits, real_profits):
    tests_data = pd.concat([pd.DataFrame({'rev': simulator_profits, 'source': 'simulator'}), 
                            pd.DataFrame({'rev': real_profits, 'source': 'true_env'})])

    alt.Chart(tests_data).mark_area(opacity=0.3, interpolate='step').encode(
                alt.X('rev', bin=True),
                alt.Y('count()', stack=None),
                alt.Color('source'))
    return