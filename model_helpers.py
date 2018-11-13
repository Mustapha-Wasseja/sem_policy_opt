import pymc3 as pm

def prep_data_for_keras_model(data, skip_y=False):
    features = ['days_before_flight', 'jetblue_demand_signal', 'jetblue_price']
    x = data[features].values
    if skip_y:
        return x
    else:
        targets = ['delta_price', 'jetblue_seats_sold', 'delta_seats_sold']
        y = [data[t].values for t in targets]
        return x, y



class WrappedPymcModel(object):
    def __init__(self,
                 model,
                 trace,
                 days_before_flight_tensor, 
                 jetblue_demand_signal_tensor, 
                 jetblue_price_tensor, 
                 delta_seats_avail_tensor, 
                 jetblue_seats_avail_tensor):
        self.model = model
        self.trace = trace
        self.days_before_flight_tensor = days_before_flight_tensor
        self.jetblue_demand_signal_tensor = jetblue_demand_signal_tensor
        self.jetblue_price_tensor = jetblue_price_tensor
        self.delta_seats_avail_tensor = delta_seats_avail_tensor
        self.jetblue_seats_avail_tensor = jetblue_seats_avail_tensor

    def predict(self, pred_X, nb_samples=1):
        self.days_before_flight_tensor.set_value([pred_X[0]])
        self.jetblue_demand_signal_tensor.set_value([pred_X[1]])
        self.jetblue_price_tensor.set_value([pred_X[2]])
        self.delta_seats_avail_tensor.set_value([pred_X[3]])
        self.jetblue_seats_avail_tensor.set_value([pred_X[4]])
        import pdb; pdb.set_trace()
        preds = pm.sample_ppc(trace=self.trace, model=self.model, samples=1, progressbar=False)
        return preds
