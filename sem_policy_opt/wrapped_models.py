from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
filterwarnings('ignore')

class WrappedModel(object):
    '''
    Base class used to wrap PyMC and Keras models (and more in future)
    Currently hard-codes info about the airline problem in predictor_names and target_names attributes
    '''
    def __init__(self, train_data, scales_data=True):
        self.train_data = train_data
        self.predictor_names = ["days_before_flight", "jb_demand_signal", "jb_price"]
        self.target_names = ['delta_price', 'jb_qty_sold', 'delta_qty_sold']
        self.n_features = len(self.predictor_names)
        self.train_X = self.train_data[self.predictor_names]
        if scales_data:
            self.scaler = StandardScaler()
            self.scaler.fit(self.train_X.as_matrix())
        self.model = self._make_fitted_model()

    def _make_fitted_model(self):
        NotImplementedError

    def predict(self, pred_data):
        NotImplementedError
    