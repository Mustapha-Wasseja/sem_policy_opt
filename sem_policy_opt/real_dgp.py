import numpy as np
import numpy as np
import pandas as pd

from numpy.random import randn

class RealDGP(object):
    def __init__(self, 
                 delta_price_fn=None, 
                 potential_customers_per_day=20, 
                 customer_level_randomness=20):
        self.potential_customers_per_day = potential_customers_per_day
        self.customer_level_randomness = customer_level_randomness
        self.delta_price_fn = delta_price_fn
        self._set_qty_demanded_fn()

    def predict(self, pred_X):
        pred_row = pred_X.iloc[0]
        delta_price = self.delta_price_fn(pred_row.delta_demand_signal, 
                                          pred_row.days_before_flight,
                                          pred_row.delta_seats_avail, 
                                          pred_row.jb_seats_avail==0)
        jb_qty_sold, delta_qty_sold = self.qty_fn(pred_row.jb_price, 
                                                  delta_price, 
                                                  pred_row.demand_level,
                                                  pred_row.jb_seats_avail, 
                                                  pred_row.delta_seats_avail)
        return {'delta_price': np.array([delta_price]),
                'jb_qty_sold': np.array([jb_qty_sold]),
                'delta_qty_sold': np.array([delta_qty_sold])}

    def _set_qty_demanded_fn(self):
        def true_calc_q(jetblue_price, delta_price, demand_level, 
                        jetblue_seats_avail, delta_seats_avail):
            jetblue_customer_level_demand = demand_level + \
                                    randn(self.potential_customers_per_day) * self.customer_level_randomness
            jetblue_consumer_surplus = jetblue_customer_level_demand - jetblue_price

            delta_customer_level_demand = demand_level + \
                                    randn(self.potential_customers_per_day) * self.customer_level_randomness
            delta_consumer_surplus = delta_customer_level_demand - delta_price

            jetblue_seats_demanded = ((jetblue_consumer_surplus > delta_consumer_surplus) * (jetblue_consumer_surplus > 0)).sum()
            delta_seats_demanded = ((delta_consumer_surplus > jetblue_consumer_surplus) * (delta_consumer_surplus > 0)).sum()
            return jetblue_seats_demanded, delta_seats_demanded
        self.qty_fn = true_calc_q