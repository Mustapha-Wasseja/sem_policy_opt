import numpy as np
import numpy as np
import pandas as pd



class CompetitiveConditions(object):
    def __init__(self, delta_price_fn=None, qty_fn=None, keras_model=None):
        self.uses_keras_model = keras_model is not None
        if self.uses_keras_model:
            self.model = keras_model
        else:
            self.delta_price_fn = delta_price_fn
            self.qty_fn = qty_fn
    
    def get_outcomes(self, jetblue_price, demand_level, jetblue_demand_signal, delta_demand_signal, days_before_flight, jetblue_seats_avail, delta_seats_avail):
        
        if self.uses_keras_model:
            preds = self.model.predict([np.array([i]) for i in [days_before_flight, jetblue_demand_signal, jetblue_price]])
            delta_price, jetblue_qty, delta_qty = (round(p[0][0]) for p in preds)
        else:       # This case of real rather than simulated market. So we use real demand_level to get quantities
            delta_price = self.delta_price_fn(delta_demand_signal, days_before_flight, delta_seats_avail, jetblue_seats_avail==0)
            jetblue_qty, delta_qty = self.qty_fn(jetblue_price, delta_price, demand_level, jetblue_seats_avail, delta_seats_avail)
        jetblue_seats_sold = np.clip(jetblue_qty, 0, jetblue_seats_avail)
        delta_seats_sold = np.clip(delta_qty, 0, delta_seats_avail)
        delta_price = max(0, delta_price)
        return delta_price, jetblue_seats_sold, delta_seats_sold

