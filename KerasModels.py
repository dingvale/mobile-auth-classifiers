from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import math

threshold = 0.5

def clear_local_variables(epoch, logs=None):
    K.get_session().run(tf.local_variables_initializer())

def recall(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.recall(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score

def precision(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.precision(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score

def false_negatives(y_true, y_pred):
    y_hat = tf.cast(y_pred > threshold, tf.float32)
    score, update = tf.metrics.false_negatives(y_true, y_hat)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update]):
        score = tf.identity(score)
    return score


def specificity(y_true, y_pred, false_negatives):
    return (sum(label == 0 for label in y_pred)[0] - false_negatives) / sum(label == 0 for label in y_true)

def logistic_model(input_n):
    print("building model")
    # This returns a tensor
    inputs = Input(shape=(input_n,))

    # a layer instance is callable on a tensor, and returns a tensor
    predictions = Dense(1, activation='sigmoid')(inputs)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall, precision, false_negatives])
    return model

def DeepNN(input_n, hidden_layers, units_per_layer=10):
    print("building model")
    # This returns a tensor
    inputs = Input(shape=(input_n,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(units_per_layer, activation='relu')(inputs)
    for _ in range(1, hidden_layers):
        x = Dense(units_per_layer, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall, precision, false_negatives])
    return model

def TrigangleDNN(input_n, hidden_layers):
    print("building model")
    # This returns a tensor
    inputs = Input(shape=(input_n,))

    neurons = math.ceil(100/2)
    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(neurons, activation='relu')(inputs)
    for _ in range(1, hidden_layers):
        neurons = math.ceil(neurons/2)
        x = Dense(neurons, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', recall, precision, false_negatives])
    return model
