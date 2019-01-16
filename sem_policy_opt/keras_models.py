import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *


def copy_and_lock_weights(src, targ, layer_names):
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


def get_keras_model(train_x, train_y, val_x, val_y, verbose=0):
    '''
    Returns a specific keras model used to predict prices and quantities. Trained on input data

    This is the "predictive" model in the airline pricing example.

    It predicts delta price from the inputs, then predicts quantities from
    the real inputs and the predicted delta_price

    Arguments:
    ----------
    train_x and val_x: list of numpy arrays. The arrays in that list correspond to
                        days_before_flight, jetblue_demand_signal, jetblue_price respectively
    train_y and val_y: list of numpy arrays. The arrays correspond to
                        delta_price, jetblue_seats_sold, delta_seats_sold respectively

    '''
    #TODO: add restore_best_weights=True as argument in es_monitor (once that change hits TensorFlow)
    es_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    batch_size = 500

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
                           batch_size=batch_size,
                           epochs=50, callbacks = [es_monitor],
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
    copy_and_lock_weights(src=delta_price_model, targ=full_model, layer_names = ['x0', 'delta_price'])

    full_model.compile(optimizer='adam',
                        loss=['mse', 'poisson', 'poisson'],
                        loss_weights=[0, 4, 1]) # 0 weight on delta_price because that part of model is already fit
    full_model.fit(x=train_x, y=train_y,
                    validation_data = [val_x, val_y],
                    epochs=50,
                    batch_size=batch_size,
                    callbacks = [es_monitor],
                    verbose=verbose)
    return full_model


def prep_for_keras_model(data, skip_y=False):
    '''
    Takes dataframe in the format produced by run_env and converts into lists of arrays
    suitable for use in get_keras_model.
    '''

    feature_names = ['days_before_flight', 'jetblue_demand_signal', 'jetblue_price']

    # if data is a list, it must be a list of only the values we want. Otherwise, it's a dataframe
    # dataframe may be a superset of desired features. If it's a list, we convert it to df and then handle on df codepath
    if type(data) is list:
        data = pd.DataFrame([data], columns=feature_names)

    x = [data[f].values for f in feature_names]
    if skip_y:
        return x
    else:
        targets = ['delta_price', 'jetblue_seats_sold', 'delta_seats_sold']
        y = [data[t].values for t in targets]
        return x, y
