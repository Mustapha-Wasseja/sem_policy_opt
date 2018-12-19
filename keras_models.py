import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *


def get_keras_model(train_x, train_y, val_x, val_y):

    days_before_flight = Input(shape=(1,))
    jb_demand_signal = Input(shape=(1,))
    jb_price = Input(shape=(1,))
    inputs_without_jb_price = Concatenate()([days_before_flight, jb_demand_signal])
    x0 = Dense(50, activation='elu')(inputs_without_jb_price)
    delta_price = Dense(1, activation='softplus', name='delta_price')(x0)

    x1 = Concatenate()([inputs_without_jb_price, delta_price, jb_price])
    x2 = Dense(50, activation='elu')(x1)

    jb_qty = Dense(1, activation='softplus', name='jb_qty')(x2)
    delta_qty = Dense(1, activation='softplus', name='delta_qty')(x2)

    keras_model = Model(inputs=[days_before_flight, jb_demand_signal, jb_price], 
                        outputs=[delta_price, jb_qty, delta_qty])
    keras_model.compile(optimizer=tf.train.AdamOptimizer(0.01), 
                        loss=['mse', 'poisson', 'poisson'], 
                        loss_weights=[1e-3, 3, 1])

    #TODO: add restore_best_weights=True as argument in es_monitor (once that change hits TensorFlow)
    es_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    keras_model.fit(x=train_x, y=train_y,
                    validation_data = [val_x, val_y],
                    steps_per_epoch = 5,
                    validation_steps = 1,
                    epochs=40,
                    callbacks = [es_monitor],
                    verbose=1)

    return keras_model


def prep_for_keras_model(data, skip_y=False):
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