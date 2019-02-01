from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
filterwarnings('ignore')

class WrappedModel(object):
    def __init__(self, train_data):
        self.train_data = train_data
        self.predictor_names = ["days_before_flight", "jb_demand_signal", "jb_price"]
        self.target_names = ['delta_price', 'jb_seats_sold', 'delta_seats_sold']
        self.n_features = len(self.predictor_names)
        self.train_X = self.train_data[self.predictor_names]
        self._model_prep()
        self._make_fitted_model()

    def _model_prep(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_X.as_matrix())

    def _make_fitted_model(self):
        NotImplementedError

    def predict(self, pred_data):
        NotImplementedError