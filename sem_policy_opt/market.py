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
    OpenAI Gym compatible implementation of the structural model for the Airline problem
    """

    def __init__(self, market_conditions, max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length):
        # TODO: Move demand_signal_noisiness into MarketConditions
        self.market_conditions = market_conditions
        self.max_demand_level = max_demand_level
        self.demand_signal_noisiness = demand_signal_noisiness
        self.seats_per_flight = seats_per_flight
        self.sales_window_length = sales_window_length
        self.__setup_for_RL()
        self.reset()

    def __setup_for_RL(self, symmetric_action_space=True):
        '''
        Set the properties of the environment (action_space and observation_space)used by Baselines RL agents

        symmetric_action_space defines whether the action space must be symmetric around 0. This is a requirement
        of some agent implementations (include Soft Actor Critic)
        '''
        if symmetric_action_space:
            low_action_space = -1*self.max_demand_level
        else:
            low_action_space = 0

        self.action_space = spaces.Box(low=low_action_space,
                                       high=self.max_demand_level,
                                       shape=(1,), dtype=np.int32)  # The price we set

        obs_space_low = np.array([0,                             # days remaining
                                  -1 * self.max_demand_level,    # demand signal
                                  0,                             # own seats remaining
                                  0])                            # int representation of competitor flight full

        obs_space_high = np.array([self.sales_window_length + 1, # days remaining
                                   self.max_demand_level + 100,  # demand signal
                                   self.seats_per_flight,        # own seats remaining
                                   1])                           # int representation of competitor flight full
        self.observation_space = spaces.Box(obs_space_low, obs_space_high, dtype=np.int32)




    def step(self, action):
        """
        The agent takes a step in the environment. Arguments and return vals follow gym API conventions

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
        jetblue_price = action
        days_before_flight = self.sales_window_length - self.current_day

        delta_price, jetblue_seats_sold, delta_seats_sold = self.market_conditions.get_outcomes(jetblue_price,
                                                                                                self.demand_level,
                                                                                                self.jetblue_demand_signal,
                                                                                                self.delta_demand_signal,
                                                                                                days_before_flight,
                                                                                                self.jetblue_seats_avail,
                                                                                                self.delta_seats_avail
                                                                                                )
        self.jetblue_seats_avail -= jetblue_seats_sold
        self.delta_seats_avail -= delta_seats_sold
        self.current_day += 1

        reward = jetblue_seats_sold * jetblue_price
        detailed_data = dict( days_before_flight = days_before_flight,
                              demand_level = self.demand_level,
                              jetblue_demand_signal = self.jetblue_demand_signal,
                              delta_demand_signal = self.delta_demand_signal,
                              jetblue_seats_avail = self.jetblue_seats_avail,
                              delta_seats_avail = self.delta_seats_avail,
                              jetblue_seats_sold = jetblue_seats_sold,
                              delta_seats_sold = delta_seats_sold,
                              jetblue_price = jetblue_price,
                              delta_price = delta_price,
                              jetblue_revenue = reward)
        self._set_next_demand_and_signals() # set before other info in each period, to allow agent to consider it
        obs = self.get_state()
        return obs, reward, self.episode_over, detailed_data

    def _set_next_demand_and_signals(self):
        self.demand_level = self.max_demand_level * np.random.rand()
        self.jetblue_demand_signal = round(self.demand_level + self.demand_signal_noisiness * np.random.randn())
        self.delta_demand_signal = round(self.demand_level + self.demand_signal_noisiness * np.random.randn())

    @property
    def episode_over(self):
        return (self.current_day == self.sales_window_length) or \
               (self.jetblue_seats_avail == 0)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_day = 0
        self.jetblue_seats_avail = self.seats_per_flight
        self.delta_seats_avail = self.seats_per_flight

        self._set_next_demand_and_signals()
        return self.get_state()

    def _render(self, mode='human', close=False):
        """This exists for compliance with Gym api. Not used in practice"""
        return

    def get_state(self):
        """Get the observation (visible state of the environment) to pass back to agent"""
        obs = [int(i) for i in
                [self.jetblue_demand_signal,
                self.sales_window_length - self.current_day,
                self.jetblue_seats_avail,
                self.delta_seats_avail == 0]]
        return obs

    def seed(self, seed):
        random.seed(seed)
        np.random.seed
