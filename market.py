# core modules
import math
import random

# 3rd party modules
from gym import spaces
import gym
import numpy as np
import pandas as pd


class Market(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, market_conditions, max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length):
        # TODO: Move demand_signal_noisiness into MarketConditions
        self.market_conditions = market_conditions
        self.max_demand_level = max_demand_level
        self.demand_signal_noisiness = demand_signal_noisiness
        self.seats_per_flight = seats_per_flight
        self.sales_window_length = sales_window_length
        self.action_space = spaces.Discrete(self.max_demand_level)  # The price we set

        # Experiment using this instead of line above
        self.observation_space = spaces.MultiDiscrete([
            (0, self.sales_window_length + 1),      # Days remaining
            # Demand signal space is larger than (0, max_demand_level) due to noisiness
            (-1 * self.max_demand_level, self.max_demand_level + 100),      # demand signal
            (0, self.seats_per_flight),             # own seats remaining
            (0, 1)                                  # competitor flight full
        ])

        obs = self.reset()


    
    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : jetblue price (will be converted to int. Ideally, comes in as int)

        Returns
        -------
        ob, reward, episode_over, info : tuple
            obs (list) :
                Current state info passed to agent. Comes from get_state
            reward (float) :
                per period profit
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                 diagnostic information for debugging.
        """
        jetblue_price = int(action)
        if self.current_day == 0:
                jetblue_seats_avail = self.seats_per_flight
                delta_seats_avail = self.seats_per_flight
        else:
            prev_day_data_dict = self.__data[-1]
            jetblue_seats_avail = prev_day_data_dict['jetblue_seats_avail'] - prev_day_data_dict['jetblue_seats_sold']
            delta_seats_avail = prev_day_data_dict['delta_seats_avail'] - prev_day_data_dict['delta_seats_sold']
        jetblue_flight_full = jetblue_seats_avail == 0
        delta_flight_full = delta_seats_avail == 0 
        days_before_flight = self.sales_window_length - self.current_day

        delta_price, jetblue_seats_sold, delta_seats_sold = self.market_conditions.get_outcomes(jetblue_price, 
                                                                                                self.demand_level, 
                                                                                                self.jetblue_demand_signal,
                                                                                                self.delta_demand_signal, 
                                                                                                days_before_flight, 
                                                                                                jetblue_seats_avail, 
                                                                                                delta_seats_avail 
                                                                                                )
        delta_price, jetblue_seats_sold, delta_seats_sold  = (self._probabilistic_rounding(i) 
                                                                for i in (delta_price, jetblue_seats_sold, delta_seats_sold))
        jetblue_revenue = jetblue_seats_sold * jetblue_price
        self.__data.append({'days_before_flight': days_before_flight, 
                           'demand_level': self.demand_level, 
                           'jetblue_demand_signal': self.jetblue_demand_signal, 
                           'delta_demand_signal': self.delta_demand_signal,
                           'jetblue_seats_avail': jetblue_seats_avail,
                           'delta_seats_avail': delta_seats_avail,
                           'jetblue_seats_sold': jetblue_seats_sold, 
                           'delta_seats_sold': delta_seats_sold,
                           'jetblue_price': jetblue_price,
                           'delta_price': delta_price,
                           'jetblue_revenue': jetblue_revenue})
        self._set_next_demand_and_signals() # set before other info in each period, to allow agent to consider it
        obs = self.get_state()
        reward = jetblue_revenue
        info = {}
        self.current_day += 1
        return obs, reward, self.episode_over, {}

    def _set_next_demand_and_signals(self):
        self.demand_level = self.max_demand_level * np.random.rand()
        self.jetblue_demand_signal = round(self.demand_level + self.demand_signal_noisiness * np.random.randn())
        self.delta_demand_signal = round(self.demand_level + self.demand_signal_noisiness * np.random.randn())

    @property
    def episode_over(self):
        return self.current_day == self.sales_window_length

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.__data = []
        self.current_day = 0
        self._set_next_demand_and_signals()
        return self.get_state()

    def _render(self, mode='human', close=False):
        return

    def get_state(self):
        """Get the observation."""
        if self.current_day > 0:
            jetblue_seats_avail = self.__data[-1]['jetblue_seats_avail']
            delta_flight_full = self.__data[-1]['delta_seats_avail'] == 0
        else:
            jetblue_seats_avail = self.seats_per_flight
            delta_flight_full = 0
                
        obs = [int(i) for i in 
                [self.jetblue_demand_signal,
                self.sales_window_length - self.current_day,
                jetblue_seats_avail,
                delta_flight_full]]
        return obs

    @property
    def data_df(self):
        assert self.episode_over
        return pd.DataFrame(self.__data)

    def _probabilistic_rounding(self, num):
        int_val = int(num)
        remainder = num - int_val
        probabilistic_remainder = np.random.binomial(n=1, p=remainder)
        return int_val + probabilistic_remainder

    def seed(self, seed):
        random.seed(seed)
        np.random.seed