import numpy as np
import pymc3 as pm
from theano import shared
from theano import tensor as tt

import os
num_cpus = os.cpu_count()

"""
Nothing in this file is in use, and in general it isn't working right now
"""

class WrappedPymcModel(object):
    def __init__(self, train_data):
        self.days_before_flight = shared(train_data.days_before_flight.values)
        self.jetblue_demand_signal = shared(train_data.jetblue_demand_signal.values)
        self.jetblue_price = shared(train_data.jetblue_price.values)
        self.delta_seats_avail = shared(train_data.delta_seats_avail.values)
        self.jetblue_seats_avail = shared(train_data.jetblue_seats_avail.values)
        self.train_data = train_data
        self._create_model()

    def _create_model(self, quicktest=False):
        with pm.Model() as pymc_model:

            purchase_prob_intercept = pm.Normal('purchase_prob_intercept', mu=0, sd=5)
            demand_signal_coeff_on_qty = pm.Normal('demand_signal_coeff_on_qty', mu=.01, sd=.1)
            price_coeff_on_qty = pm.Normal('price_coeff_on_qty', mu=-0.01, sd=.1)
            cross_price_coeff_on_qty = pm.Normal('cross_price_coeff_on_qty', mu=0.01, sd=.1)

            jb_purchase_prob = pm.math.sigmoid(purchase_prob_intercept +  \
                                        self.jetblue_demand_signal * demand_signal_coeff_on_qty + \
                                        self.jetblue_price * price_coeff_on_qty)

            cross_signal_coeff_on_qty = pm.Normal('cross_signal_coeff_on_qty', mu=0.01, sd=.1)

            delta_purchase_prob = pm.math.sigmoid(purchase_prob_intercept +  \
                                        self.jetblue_demand_signal * cross_signal_coeff_on_qty + \
                                        self.jetblue_price * cross_price_coeff_on_qty)

            max_seats_sold = max(self.train_data.jetblue_seats_sold.max(),
                                 self.train_data.delta_seats_sold.max())
            potential_customers = pm.DiscreteUniform('potential_customers',
                                                     max_seats_sold, 5 * max_seats_sold)
            jb_qty = pm.Binomial('jb_qty',
                                 n=potential_customers,
                                 p=jb_purchase_prob,
                                 observed=self.train_data.jetblue_seats_sold.values)
            delta_qty = pm.Binomial('delta_qty',
                                    n=potential_customers,
                                    p=delta_purchase_prob,
                                    observed=self.train_data.delta_seats_sold.values)

            if quicktest:
                    n_samples=100
                    n_tune=100
            else:
                    n_samples = 1000
                    n_tune = 1000
            self.trace = pm.sample(n_samples, tune=n_tune, cores=num_cpus, init='adapt_diag')
            self.model = pymc_model
            return

    def predict(self, pred_X, nb_samples=1):
        if type(pred_X) == list:
            self.days_before_flight.set_value(pred_X[0])
            self.jetblue_demand_signal.set_value(pred_X[1])
            self.jetblue_price.set_value(pred_X[2])

        preds = pm.sample_ppc(trace=self.trace, model=self.model, samples=1, progressbar=False)
        missing_delta_price_preds = np.full_like(preds['jb_qty'], np.nan)
        out = np.vstack([missing_delta_price_preds,
                         preds['jb_qty'],
                         preds['delta_qty']])
        return out
