from keras.layers import Input, BatchNormalization, Dropout, Activation, Dense
from keras.models import Model

"""
Simple dense network to test with web frameworks.
"""
def dense_ecg_net(input_shape=(360,), cycles=3):
    a = x = Input(shape=input_shape, name='input')

    for i in range(1, cycles + 1):
        x = Dense(units=90)(x)
        x = BatchNormalization(name='bn_%s' % i)(x)
        x = Activation(activation='relu', name='act_%s' % i)(x)
        x = Dropout(rate=0.2)(x)

    x = Dense(units=4, activation='softmax', name='dense')(x)

    return Model(inputs=a, outputs=x, name='dense_ecg_net')
