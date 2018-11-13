import numpy as np
import pandas as pd
from model_helpers import prep_data_for_keras_model



class Market(object):
    def __init__(self):
        self.flights_simulated = 0
        
        data_colnames = ['flight_id', 'days_before_flight', 'average_demand',
                         'jetblue_demand_signal', 'delta_demand_signal',  
                         'jetblue_seats_avail', 'delta_seats_avail',
                         'jetblue_seats_sold', 'delta_seats_sold', 
                         'jetblue_price', 'delta_price',
                         'jetblue_revenue']
        self._data = pd.DataFrame([], columns= data_colnames)
    
    def simulate_flight_sales(self):
        this_flight_data = pd.DataFrame([], columns=self._data.columns)
        
        for day in range(self.sales_window_length):
            if day == 0:
                jetblue_seats_avail = self.seats_per_flight
                delta_seats_avail = self.seats_per_flight
            else:
                prev_day = this_flight_data.iloc[-1]
                jetblue_seats_avail = prev_day.jetblue_seats_avail - prev_day.jetblue_seats_sold
                delta_seats_avail = prev_day.delta_seats_avail - prev_day.delta_seats_sold
            jetblue_flight_full = jetblue_seats_avail == 0
            delta_flight_full = delta_seats_avail == 0 
            days_before_flight = self.sales_window_length - day
            average_demand = self.max_demand_level * np.random.rand()
            jetblue_demand_signal = np.clip(average_demand + self.demand_signal_noisiness * np.random.randn(), 0, self.max_demand_level)
            delta_demand_signal = np.clip(average_demand + self.demand_signal_noisiness * np.random.randn(), 0, self.max_demand_level)
            jetblue_price, delta_price, jetblue_seats_sold, delta_seats_sold = self.get_p_and_q(average_demand, days_before_flight,
                                                                                                jetblue_demand_signal, delta_demand_signal, 
                                                                                                jetblue_seats_avail, delta_seats_avail
                                                                                                )
            jetblue_revenue = jetblue_seats_sold * jetblue_price
            this_flight_data.loc[day] = [self.flights_simulated, days_before_flight, 
                                         average_demand, 
                                         jetblue_demand_signal, delta_demand_signal,  
                                         jetblue_seats_avail, delta_seats_avail, 
                                         jetblue_seats_sold, delta_seats_sold, 
                                         jetblue_price, delta_price,
                                         jetblue_revenue]
        self.flights_simulated += 1

        return this_flight_data


    def calc_seats_sold(self, jetblue_price, delta_price, demand_level, jetblue_seats_avail, delta_seats_avail):

        return np.clip(jetblue_seats_demanded, 0, jetblue_seats_avail), \
               np.clip(delta_seats_demanded, 0, delta_seats_avail)
        

class RealMarket(Market):
    def __init__(self, max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length, jetblue_price_fn, delta_price_fn, potential_customers_per_day, customer_level_randomness):
        self.max_demand_level = max_demand_level
        self.demand_signal_noisiness = demand_signal_noisiness
        self.jetblue_price_fn = jetblue_price_fn
        self.delta_price_fn = delta_price_fn
        self.seats_per_flight = seats_per_flight
        self.sales_window_length = sales_window_length
        self.potential_customers_per_day = potential_customers_per_day
        self.customer_level_demand_randomness = customer_level_randomness
        super().__init__()

    def get_p_and_q(self, demand_level, days_before_flight, jetblue_demand_signal, delta_demand_signal, jetblue_seats_avail, delta_seats_avail):
        jetblue_price = self.jetblue_price_fn(jetblue_demand_signal, delta_demand_signal, days_before_flight, jetblue_seats_avail, delta_seats_avail)
        delta_price = self.delta_price_fn(delta_demand_signal, jetblue_demand_signal, days_before_flight, delta_seats_avail, jetblue_seats_avail)

        jetblue_customer_level_demand = demand_level + np.random.randn(self.potential_customers_per_day) * 20
        jetblue_consumer_surplus = jetblue_customer_level_demand - jetblue_price

        delta_customer_level_demand = demand_level + np.random.randn(self.potential_customers_per_day) * 20
        delta_consumer_surplus = delta_customer_level_demand - delta_price

        jetblue_seats_demanded = ((jetblue_consumer_surplus > delta_consumer_surplus) * (jetblue_consumer_surplus > 0)).sum()
        delta_seats_demanded = ((delta_consumer_surplus > jetblue_consumer_surplus) * (delta_consumer_surplus > 0)).sum()
        jetblue_seats_sold = np.clip(jetblue_seats_demanded, 0, jetblue_seats_avail)
        delta_seats_sold = np.clip(delta_seats_demanded, 0, delta_seats_avail)

        return jetblue_price, delta_price, jetblue_seats_sold, delta_seats_sold


class SimulatedMarket(Market):
    def __init__(self, max_demand_level, demand_signal_noisiness, seats_per_flight, sales_window_length, jetblue_price_fn, q_and_delta_price_model):
        self.max_demand_level = max_demand_level
        self.demand_signal_noisiness = demand_signal_noisiness
        self.seats_per_flight = seats_per_flight
        self.sales_window_length = sales_window_length
        self.jetblue_price_fn = jetblue_price_fn
        self.q_and_delta_price_model = q_and_delta_price_model
        self.uses_keras_model = ('layers' in dir(q_and_delta_price_model)) and (type(q_and_delta_price_model.layers) == list)     # hacky way to check this
        super().__init__()
    
    def get_p_and_q(self, demand_level, days_before_flight, jetblue_demand_signal, delta_demand_signal, jetblue_seats_avail, delta_seats_avail):
        jetblue_price = self.jetblue_price_fn(jetblue_demand_signal, delta_demand_signal, days_before_flight, jetblue_seats_avail, delta_seats_avail)
        if self.uses_keras_model:
            pred_data = prep_data_for_keras_model(pd.DataFrame({'days_before_flight': [days_before_flight],
                                                                'jetblue_demand_signal': [jetblue_demand_signal],
                                                                'jetblue_price': [jetblue_price]}),
                                                 skip_y=True)
            preds = self.q_and_delta_price_model.predict(pred_data)
            delta_price, jetblue_seats_sold, delta_seats_sold = (round(p[0][0]) for p in preds)
        
        jetblue_seats_sold = np.clip(jetblue_seats_sold, 0, jetblue_seats_avail)
        delta_seats_sold = np.clip(delta_seats_sold, 0, delta_seats_avail)
        return jetblue_price, delta_price, jetblue_seats_sold, delta_seats_sold
