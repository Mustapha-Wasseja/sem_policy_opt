import numpy as np
from numpy.random import randn
import pandas as pd
import pymc3 as pm
from pymc3.math import dot, tanh
import theano
from theano import shared
from theano import tensor as tt
from theano.tensor.nnet.nnet import relu, elu, softplus
from sklearn.preprocessing import StandardScaler
from sem_policy_opt.wrapped_models import WrappedModel
from warnings import filterwarnings
filterwarnings('ignore')

class WrappedPymcModel(WrappedModel):
    def _make_fitted_model(self, n_hidden=10):

        self.model_input = shared(self._prep_X(self.train_X))
        self.jb_qty_sold_output = shared(self.train_data.jb_qty_sold.values)
        self.delta_qty_sold_output = shared(self.train_data.delta_qty_sold.values)
        self.delta_price = shared(self.train_data.delta_price.values)

        # Initialize random weights
        init_1 = randn(self.n_features, n_hidden)
        init_2 = randn(n_hidden, n_hidden)
        init_jb_sold_out = randn(n_hidden)
        init_delta_sold_out = randn(n_hidden)
        init_delta_price_out = randn(n_hidden)

        with pm.Model() as neural_network:
            # Weights from input to hidden layer
            weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                     shape=(self.n_features, n_hidden),
                                     testval=init_1)

            # Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                    shape=(n_hidden, n_hidden),
                                    testval=init_2)

            # Weights from hidden layer to output
            weights_jb_out = pm.Normal('w_jb_out', 0, sd=1,
                                       shape=(n_hidden,),
                                       testval=init_jb_sold_out)

            weights_delta_out = pm.Normal('w_delta_out', 0, sd=1,
                                          shape=(n_hidden,),
                                          testval=init_delta_sold_out)

            weights_delta_price = pm.Normal('w_delta_price', 0, sd=100,
                                          shape=(n_hidden,),
                                          testval=init_delta_price_out)

            act_1 = relu(dot(self.model_input, weights_in_1))
            act_2 = relu(dot(act_1, weights_1_2))

            jb_sold_intercept = pm.Normal('jb_sold_intercept', 0, sd=10)
            delta_sold_intercept = pm.Normal('delta_sold_intercept', 0, sd=10)
            delta_price_intercept = pm.Normal('delta_price_intercept', 0, sd=100)

            jb_sold_lambda = softplus(dot(act_2, weights_jb_out) + jb_sold_intercept)
            delta_sold_lambda = softplus(dot(act_2, weights_delta_out) + delta_sold_intercept)
            delta_price_mu = dot(act_2, weights_delta_price)
            
            # outputs
            jb_qty_node = pm.Poisson('jb_qty_sold',
                                 jb_sold_lambda,
                                 observed=self.jb_qty_sold_output
                                 )
            delta_qty_node = pm.Poisson('delta_qty_sold',
                                    delta_sold_lambda,
                                    observed=self.delta_qty_sold_output
                                    )
            delta_price_node = pm.Normal('delta_price',
                                    delta_price_mu,
                                    # Use large SD as hack to give price preds lower "weight" than qty
                                    sd=100,     
                                    observed=self.delta_price)

        with neural_network:
            inference = pm.ADVI()
            self.approx = pm.fit(n=35000, method=inference)
        self.pred_fns = self._get_pred_fns(neural_network)
        return neural_network

    def _get_pred_fns(self, model):
        '''Create function to get predictions for each observedRV is in model.
        This allows us to ignore some intricacies of theano/PyMC elsewhere.

        For speed, works at theano level rather than calling PyMC's sample_ppc
        '''
        def get_conditional_poisson_param_fn(var):
            # set up placeholder tensors
            x = tt.matrix('X')
            x.tag.test_value = self.train_X.values

            pred_graph = self.approx.sample_node(var.distribution.mu,
                                                 more_replacements={self.model_input: x})
            return theano.function([x], pred_graph)
        return {var.name: get_conditional_poisson_param_fn(var) 
                                for var in model.observed_RVs}

    def _prep_X(self, data):
        out_data = data[self.predictor_names].values
        return self.scaler.transform(out_data)

    def predict(self, pred_X, nb_samples=1):
        transformed_data = self._prep_X(pred_X)
        preds = {targ_name: pred_fn(transformed_data) 
                                for targ_name, pred_fn in self.pred_fns.items()}
        return preds
