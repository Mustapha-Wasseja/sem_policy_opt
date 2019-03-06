import random
from gym import spaces
import gym
import numpy as np
import pandas as pd


class Market(gym.Env):
    '''
    OpenAI Gym compatible implementation of the structural model for Airline problem
    '''

    def __init__(self, model, market_details):
        self.model = model
        self.max_demand_level = market_details['max_demand_level']
        self.demand_signal_noisiness = market_details['demand_signal_noisiness']
        self.seats_per_flight = market_details['seats_per_flight']
        self.sales_window_length = market_details['sales_window_length']
        self.__setup_for_RL()
        self.reset()

    def __setup_for_RL(self, symmetric_action_space=True):
        '''
        Set the properties of the environment (action_space and observation_space)
        used by Baselines RL agents

        symmetric_action_space defines whether the action space must be symmetric 
        around 0. This is a requirement of some agent implementations 
        (include Soft Actor Critic)
        '''
        if symmetric_action_space:
            low_action_space = -1*self.max_demand_level
        else:
            low_action_space = 0

        # The price we set
        self.action_space = spaces.Box(low=low_action_space,
                                       high=self.max_demand_level,
                                       shape=(1,), dtype=np.int32)  

        obs_space_low = np.array([0,                             # days remaining
                                  -1 * self.max_demand_level,    # demand signal
                                  0,                             # own seats remaining
                                  0])                            # competitor flight full (int)

        obs_space_high = np.array([self.sales_window_length + 1, # days remaining
                                   self.max_demand_level + 100,  # demand signal
                                   self.seats_per_flight,        # own seats remaining
                                   1])                           # competitor flight full (int)
        self.observation_space = spaces.Box(obs_space_low, obs_space_high, dtype=np.int32)




    def step(self, action):
        '''
        The agent takes a step in the environment. 
        Arguments and return vals follow gym API conventions

        Parameters
        ----------
        action : jetblue price (will be converted to int.)

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
        '''
        jb_price = action
        days_before_flight = self.sales_window_length - self.current_day

        prediction_data = pd.DataFrame(dict(days_before_flight=[days_before_flight],
                                            jb_demand_signal=[self.jb_demand_signal],
                                            jb_price=[jb_price],
                                            delta_demand_signal=[self.delta_demand_signal],
                                            delta_seats_avail=[self.delta_seats_avail],
                                            jb_seats_avail=[self.jb_seats_avail],
                                            demand_level=[self.demand_level]))

        raw_preds = self.model.predict(prediction_data)
        upper_limit = {'delta_price': np.inf,
                        'jb_qty_sold': self.jb_seats_avail,
                        'delta_qty_sold': self.delta_seats_avail}
        
        assert all(a.shape==(1,) for a in raw_preds.values())
        bound_pred = lambda x: self._probabilistic_round(np.clip(raw_preds[x][0], 
                                                                 0, 
                                                                 upper_limit[x]))
        delta_price = bound_pred('delta_price')
        jb_qty_sold = bound_pred('jb_qty_sold')
        delta_qty_sold = bound_pred('delta_qty_sold')
        
        self.jb_seats_avail -= jb_qty_sold
        self.delta_seats_avail -= delta_qty_sold
        self.current_day += 1

        reward = jb_qty_sold * jb_price
        detailed_data = dict( days_before_flight = days_before_flight,
                              demand_level = self.demand_level,
                              jb_demand_signal = self.jb_demand_signal,
                              delta_demand_signal = self.delta_demand_signal,
                              jb_seats_avail = self.jb_seats_avail,
                              delta_seats_avail = self.delta_seats_avail,
                              jb_qty_sold = jb_qty_sold,
                              delta_qty_sold = delta_qty_sold,
                              jb_price = jb_price,
                              delta_price = delta_price,
                              jb_revenue = reward)
        # set before other info in each period, to allow agent to consider it
        self._set_next_demand_and_signals() 
        obs = self.get_state()
        return obs, reward, self.episode_over, detailed_data

    def _set_next_demand_and_signals(self):
        self.demand_level = self.max_demand_level * np.random.rand()
        self.jb_demand_signal = round(self.demand_level + \
                                      self.demand_signal_noisiness * np.random.randn())
        self.delta_demand_signal = round(self.demand_level + \
                                         self.demand_signal_noisiness * np.random.randn())

    @property
    def episode_over(self):
        return (self.current_day == self.sales_window_length) or \
               (self.jb_seats_avail == 0)

    def reset(self):
        '''
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        '''
        self.current_day = 0
        self.jb_seats_avail = self.seats_per_flight
        self.delta_seats_avail = self.seats_per_flight

        self._set_next_demand_and_signals()
        return self.get_state()

    def _render(self, mode='human', close=False):
        '''This exists for compliance with Gym api. Not used in practice'''
        return

    def get_state(self):
        '''Get the observation (visible state of the environment) to pass back to agent'''
        obs = [int(i) for i in
                [self.jb_demand_signal,
                self.sales_window_length - self.current_day,
                self.jb_seats_avail,
                self.delta_seats_avail == 0]]
        return obs

    def seed(self, seed):
        random.seed(seed)


    def _probabilistic_round(self, num):
        '''
        Converts float values to integers in way that is smooth in expectation
        3.8 is converted to 3 with probability 20% and to value 4 with probability 80%
        '''
        int_val = int(num)
        remainder = num - int_val
        probabilistic_remainder = np.random.binomial(n=1, p=remainder)
        return int_val + probabilistic_remainder