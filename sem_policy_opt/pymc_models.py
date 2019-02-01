import numpy as np
from numpy.random import randn
import pandas as pd
import pymc3 as pm
from pymc3.math import dot
import theano
from theano import shared
from theano import tensor as tt
from theano.tensor.nnet.nnet import relu, elu, softplus
from sklearn.preprocessing import StandardScaler
from sem_policy_opt.wrapped_models import WrappedModel
from warnings import filterwarnings
filterwarnings('ignore')

class WrappedPymcModel(WrappedModel):
    def _make_fitted_model(self, n_hidden=5, trace_size=1):

        self.model_input = shared(self._prep_X(self.train_X))
        self.jb_qty_sold_output = shared(self.train_data.jb_qty_sold.values)
        self.delta_qty_sold_output = shared(self.train_data.delta_qty_sold.values)

        # Initialize random weights
        init_1 = randn(self.n_features, n_hidden)
        init_2 = randn(n_hidden, n_hidden)
        init_jb_sold_out = randn(n_hidden)
        init_delta_sold_out = randn(n_hidden)

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

            act_1 = relu(dot(self.model_input, weights_in_1))
            act_2 = relu(dot(act_1, weights_1_2))
            jb_sold_lambda = softplus(dot(act_2, weights_jb_out))
            delta_sold_lambda = softplus(dot(act_2, weights_delta_out))

            # outputs
            self.jb_qty_node = pm.Poisson('jb_qty_sold',
                                 jb_sold_lambda,
                                 observed=self.jb_qty_sold_output
                                 )
            self.delta_qty_node = pm.Poisson('delta_qty_sold',
                                    delta_sold_lambda,
                                    observed=self.delta_qty_sold_output
                                    )

        with neural_network:
            inference = pm.SVGD(n_particles=500, jitter=1)
            self.approx = inference.approx

        self.model = neural_network
        #self.trace = self.approx.sample(draws=trace_size)

    def _prep_X(self, data):
        data = data[self.predictor_names]
        return self.scaler.transform(data)
        
    def predict(self, pred_X, nb_samples=1):
        if type(pred_X) == list:
            # These would be a single dimensional array for each variable.
            # Convert those to standard DF format where each var is a column
            assert all(a.ndim==1 for a in pred_X)
            pred_X = pd.DataFrame(data=np.hstack([i[:, np.newaxis] for i in pred_X]),
                                 columns=self.predictor_names)
        n_rows = pred_X.shape[0]
        transformed_data = self._prep_X(pred_X)
        import pdb; pdb.set_trace()
        
        with self.model:
            jb_qty_preds = self.approx.sample_node(self.jb_qty_node, 
                                                   more_replacements={self.model_input: transformed_data}).eval()
            delta_qty_preds = self.approx.sample_node(self.delta_qty_node,
                                                      more_replacements={self.model_input: transformed_data}).eval()
        preds = {'jb_qty_sold': jb_qty_preds,
                 'delta_qty_sold': delta_qty_preds,
                 'delta_price': np.full_like(jb_qty_preds, np.nan)}
        print(pred_X.shape)
        print(preds['jb_qty_sold'].shape)
        return preds
