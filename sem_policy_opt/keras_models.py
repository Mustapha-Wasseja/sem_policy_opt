import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from sem_policy_opt.wrapped_models import WrappedModel

class WrappedKerasModel(WrappedModel):

    def _copy_and_lock_weights(src, targ, layer_names):
        '''
        Copy weights from layers named in layer_names from src_model to targ_model layers of same name
        Then set trainable of targ_model layers to false

        This can be used for layers leading to some intermediate outcome, where data for the
        intermediate outcome was available historically (and could be used as a target) but
        where data will not be available in the future when this is used to predict final outcomes.
        The src model is where the intermediate variable is the outcome, and that model has been trained
        The target model is where we want to treat layers leading up to the intermediate variable as fixed.

        Arguments:
        ----------
        src: trained keras model
        targ: keras model we want to copy weights to and then set those layers as not trainable
        layer_names: list of strings of list of layer_names for which we want to copy weights
        '''

        for name in layer_names:
            src_layer = [l for l in src.layers if l.name == name][0]
            targ_layer = [l for l in targ.layers if l.name == name][0]
            targ_layer.set_weights(src_layer.get_weights())
            targ_layer.trainable = False


    def _make_fitted_model(self, verbose=0):
        '''
        Creates and sets the predictive model and the WrappedModel (using self.train_data)
        '''

        train_x = self._prep_X(self.train_data)
        train_y = self._prep_y(self.train_data)
        # TODO: Add early stopping back after figuring out better way to build in validation to
        #       keras models without negatively affecting interface for PyMC models
        # es_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')


        days_before_flight = Input(shape=(1,))
        jb_demand_signal = Input(shape=(1,))
        jb_price = Input(shape=(1,))
        inputs_without_jb_price = Concatenate()([days_before_flight, jb_demand_signal])
        x0 = Dense(100, activation='elu', name='x0')(inputs_without_jb_price)
        delta_price = Dense(1, activation='softplus', name='delta_price')(x0)

        # Build just the part of the model that predicts delta_price. This doesn't include jetblue_price
        # as an input because delta can't see that.
        delta_price_model = Model(inputs=[days_before_flight, jb_demand_signal],
                                outputs=[delta_price])
        delta_price_model.compile(optimizer='adam',
                                loss='mse')
        delta_price_model.fit( x=train_x[:-1],
                            y=train_y[0],
                            validation_data = [val_x[:-1], val_y[0]],
                            batch_size=500,
                            epochs=50, 
                            # callbacks = [es_monitor],
                            verbose=verbose)

        # early versions had positive cross-demand elasticities. Add price_diff to encourage model
        # to view direct effect of jb price, and outweigh it's use as proxy for demand
        price_diff = Subtract()([jb_price, delta_price])
        qty_predictors = Concatenate()([inputs_without_jb_price, delta_price, jb_price, price_diff])

        x1 = Dense(100, activation='elu')(qty_predictors)
        x2 = Dropout(0.5)(x1)
        x3 = Dense(100, activation='elu')(x2)
        jb_qty = Dense(1, activation='softplus', name='jb_qty')(x3)

        delta_qty = Dense(1, activation='softplus', name='delta_qty')(x3)

        full_model = Model(inputs=[days_before_flight, jb_demand_signal, jb_price],
                            outputs=[delta_price, jb_qty, delta_qty])


        # Freeze layers that predict delta_price and train model to predict quantities
        # Setting layers to trainable is tricky because they change identity when put into a model
        # so delta_price.trainable = False loses the trainable property and delta_model is not in full_model.layers
        _copy_and_lock_weights(src=delta_price_model, targ=full_model, layer_names = ['x0', 'delta_price'])

        full_model.compile(optimizer='adam',
                            loss=['mse', 'poisson', 'poisson'],
                            loss_weights=[0, 4, 1]) # 0 weight on delta_price because that part of model is already fit
        full_model.fit(x=train_x, y=train_y,
                        epochs=50,
                        batch_size=batch_size,
                        #callbacks = [es_monitor],
                        verbose=verbose)
        self.model = full_model

    def predict(self, pred_data):
        assert type(pred_data) == list
        df = pd.DataFrame([data], columns=feature_names)
        processed_data = self._prep_X(df)
        preds = self.model.predict(processed_data)
        out = dict(zip(self.target_names, preds))
        return preds

    def _prep_X(self, data):
        '''
        Takes dataframe in the format produced by run_env and converts into lists of arrays
        suitable for use in keras model with multiple inputs
        '''
        return [data[f].values for f in feature_names]

    def _prep_y(self, data):
        return [data[t].values for t in self.target_names]
