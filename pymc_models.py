import pymc3 as pm
from theano import shared
from theano import tensor as tt

import os
num_cpus = os.cpu_count()



class WrappedPymcModel(object):
    def __init__(self, train_data):
        self.days_before_flight = shared(train_data.days_before_flight.values)
        self.jetblue_demand_signal = shared(train_data.jetblue_demand_signal.values)
        self.jetblue_price = shared(train_data.jetblue_price.values)
        self.delta_seats_avail = shared(train_data.delta_seats_avail.values)
        self.jetblue_seats_avail = shared(train_data.jetblue_seats_avail.values)
        self.train_data = train_data
        self._create_model()
    
    def _create_model(self, quicktest=True):
        with pm.Model() as pymc_model:
            
            delta_price_intercept = pm.Normal('delta_price_intercept', mu=0, sd=100)
            cross_demand_signal_coeff_on_price = pm.Normal('cross_demand_signal_coeff_on_price', mu=.5, sd=5)
            seats_remaining_coeff_on_price = pm.Normal('seats_remaining_coeff_on_price', mu=-1, sd=5)
            days_before_coeff_on_price = pm.Normal('days_before_coeff_on_price', mu=0, sd=5)

            delta_full = tt.eq(self.delta_seats_avail, 0).astype('int8')
            jb_full = tt.eq(self.jetblue_seats_avail, 0).astype('int8')

            delta_price_mu = delta_price_intercept + \
                                cross_demand_signal_coeff_on_price * self.jetblue_demand_signal + \
                                seats_remaining_coeff_on_price * self.delta_seats_avail + \
                                days_before_coeff_on_price * self.days_before_flight

            delta_price_sd = pm.Gamma('delta_price_sd', mu=50, sd=100)
            delta_price = pm.Normal('delta_price', mu = delta_price_mu, sd=delta_price_sd, observed=self.train_data.delta_price.values)

            purchase_prob_intercept = pm.Normal('purchase_prob_intercept', mu=-5, sd=5)
            demand_signal_coeff_on_qty = pm.Normal('demand_signal_coeff_on_qty', mu=.001, sd=.05)
            days_before_coeff_on_qty = pm.Normal('days_before_coeff_on_qty', mu=0, sd=.2)
            price_multiplier_on_qty = pm.Normal('price_multiplier_on_qty', mu=-0.01, sd=.2)
            cross_price_multiplier_on_qty = pm.Normal('cross_price_multiplier_on_qty', mu=0.005, sd=.1)

            jb_purchase_prob = pm.math.sigmoid(purchase_prob_intercept +  \
                                        self.jetblue_demand_signal * demand_signal_coeff_on_qty + \
                                        self.days_before_flight * days_before_coeff_on_qty + \
                                        self.jetblue_price * price_multiplier_on_qty + \
                                        delta_price * cross_price_multiplier_on_qty * delta_full)
    
            delta_purchase_prob = pm.math.sigmoid(purchase_prob_intercept +  \
                                        self.jetblue_demand_signal * demand_signal_coeff_on_qty + \
                                        self.days_before_flight * days_before_coeff_on_qty + \
                                        delta_price * price_multiplier_on_qty + \
                                        self.jetblue_price * cross_price_multiplier_on_qty * jb_full)

            max_seats_sold = max(self.train_data.jetblue_seats_sold.max(), self.train_data.delta_seats_sold.max())
            potential_customers = pm.DiscreteUniform('potential_customers', max_seats_sold, 3 * max_seats_sold)
    
            # Include (1-full) to account for inability to sell on full flights.  Doesn't handle partially limited sales well
            jb_qty = pm.Binomial('jb_qty', n=potential_customers * (1-jb_full), p=jb_purchase_prob, observed=self.train_data.jetblue_seats_sold.values)
            delta_qty = pm.Binomial('delta_qty', n=potential_customers * (1-delta_full), p=delta_purchase_prob, observed=self.train_data.delta_seats_sold.values)

            if quicktest:
                    n_samples=10
                    n_tune=15
            else:
                    n_samples = 1000
                    n_tune = 1000
            self.trace = pm.sample(n_samples, tune=n_tune, cores=num_cpus)
            self.model = pymc_model
            return

    def predict(self, pred_X, nb_samples=1):
        self.days_before_flight.set_value([pred_X[0]])
        self.jetblue_demand_signal.set_value([pred_X[1]])
        self.jetblue_price.set_value([pred_X[2]])
        self.delta_seats_avail.set_value([pred_X[3]])
        self.jetblue_seats_avail.set_value([pred_X[4]])
        preds = pm.sample_ppc(trace=self.trace, model=self.model, samples=1, progressbar=False)
        return preds