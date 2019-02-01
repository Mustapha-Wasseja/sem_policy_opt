import numpy as np
import numpy as np
import pandas as pd

class CompetitiveConditions(object):
    '''
    The process that determines competitor price and quantities sold.

    This can wrap either the true data generating process or a predictive model of that process

    '''
    def __init__(self, delta_price_fn=None, qty_fn=None, predictive_model=None):
        self.model_based_simulation = predictive_model is not None
        if self.model_based_simulation:
            self.model = predictive_model
        else:
            self.delta_price_fn = delta_price_fn
            self.qty_fn = qty_fn

    def get_outcomes(self, jb_price, demand_level, jb_demand_signal,
                    delta_demand_signal, days_before_flight, jb_seats_avail,
                    delta_seats_avail):

        if self.model_based_simulation:
            prediction_data = pd.DataFrame(dict(days_before_flight=[days_before_flight],
                                                jb_demand_signal=[jb_demand_signal],
                                                jb_price=[jb_price]))
            preds = self.model.predict(prediction_data)
            delta_price, jb_qty, delta_qty = (preds[i][0] for i in ['delta_price', 'jb_qty_sold', 'delta_qty_sold'])
        else:       # Case of real market. Use real demand_level and real dgp to get quantities
            delta_price = self.delta_price_fn(delta_demand_signal, days_before_flight,
                                              delta_seats_avail, jb_seats_avail==0)
            jb_qty, delta_qty = self.qty_fn(jb_price, delta_price, demand_level,
                                                 jb_seats_avail, delta_seats_avail)

        jb_qty_sold = np.clip(jb_qty, 0, jb_seats_avail)
        delta_qty_sold = np.clip(delta_qty, 0, delta_seats_avail)
        delta_price = np.clip(delta_price, 0, np.inf)
        #jb_qty_sold, delta_qty_sold, delta_price = (self._probabilistic_rounding(i)
        #                                                  for i in [jb_qty_sold, delta_qty_sold, delta_price])
        return delta_price, jb_qty_sold, delta_qty_sold

    def _probabilistic_rounding(self, num):
        '''
        Converts float values to integers in way that is smooth in expectation

        3.8 is converted to 3 with probability 20% and to value 4 with probability 80%
        '''
        int_val = int(num)
        remainder = num - int_val
        probabilistic_remainder = np.random.binomial(n=1, p=remainder)
        return int_val + probabilistic_remainder