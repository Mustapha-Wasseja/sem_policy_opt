import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *

from model_helpers import prep_data_for_keras_model

def get_keras_model(train_x, train_y, val_x, val_y):

    num_features = train_x.shape[1]

    inputs = Input(shape=(num_features,))
    x = Dense(50, activation='elu')(inputs)
    x1 = Dense(10, activation='elu')(x)
    x2 = Dense(10, activation='elu')(x)
    x3 = Dense(10, activation='elu')(x)
    delta_price = Dense(1)(x1)
    # TODO: Let delta_price affect quantities
    jb_qty = Dense(1)(x2)
    delta_qty = Dense(1)(x3)

    keras_model = Model(inputs=inputs, outputs=[delta_price, jb_qty, delta_qty])
    keras_model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', loss_weights=[1e-3, 2, 1])

    #TODO: add restore_best_weights=True as argument in es_monitor (once that change hits TensorFlow)
    es_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
    keras_model.fit(x=train_x, y=train_y, 
                    validation_data = [val_x, val_y],
                    steps_per_epoch = 3,
                    validation_steps = 1,
                    epochs=50,
                    callbacks = [es_monitor],
                    verbose=0)

    return keras_model


def prep_data_for_keras_model(data, skip_y=False):
    features = ['days_before_flight', 'jetblue_demand_signal', 'jetblue_price']
    x = data[features].values
    if skip_y:
        return x
    else:
        targets = ['delta_price', 'jetblue_seats_sold', 'delta_seats_sold']
        y = [data[t].values for t in targets]
        return x, y