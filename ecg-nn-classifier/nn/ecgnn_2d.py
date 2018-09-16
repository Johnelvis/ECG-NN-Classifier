from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation, Dense, MaxPool2D, Add, Flatten, ZeroPadding2D, Permute
from keras.models import Model

"""
ECG classification net based on "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks", extended
to use multiple ecg data streams. (https://arxiv.org/pdf/1707.01836.pdf) by Pranav Rajpurkar et al.
"""
def stanford_net_2d(input_shape=(360, 2, 1), res_layers=16):
    assert res_layers & 0x1 == 0, 'Amount of residual layers should be even'

    prev_output_size, output_size = 64, 64

    a = Input(shape=input_shape, name='input')
    x = Conv2D(filters=64, kernel_size=16, strides=1, padding='same', kernel_initializer='he_uniform', name='conv1')(a)
    x = BatchNormalization(name='bn1')(x)
    x = Activation(activation='relu', name='act1')(x)

    shrt = MaxPool2D(pool_size=2, strides=2, name='mp1', padding='same')(x)

    x = Conv2D(filters=64, kernel_size=16, strides=2, padding='same', kernel_initializer='he_uniform', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation(activation='relu', name='act2')(x)
    x = Dropout(rate=0.2, name='dr1')(x)
    x = Conv2D(filters=64, kernel_size=16, strides=1, padding='same', kernel_initializer='he_uniform', name='conv3')(x)

    x = Add()([shrt, x])

    for i in range(0, res_layers):
        output_size = 64 * ((i >> 2) + 1)

        shrt = MaxPool2D(pool_size=2, strides=(2 if i % 2 == 0 else 1), name='Block_%s/mp2' % i, padding='same')(x)

        if prev_output_size < output_size:
            shrt = Permute((1, 3, 2))(shrt)
            shrt = ZeroPadding2D(padding=((0, 0), (0, 64)))(shrt)
            shrt = Permute((1, 3, 2))(shrt)

        x = BatchNormalization(name='Block_%s/bn2' % i)(x)
        x = Activation(activation='relu', name='Block_%s/act3' % i)(x)
        x = Dropout(rate=0.2)(x)
        x = Conv2D(filters=output_size, kernel_size=16, strides=(2 if i % 2 == 0 else 1), padding='same', kernel_initializer='he_uniform', name='Block_%s/conv4' % i)(x)
        x = BatchNormalization(name='Block_%s/bn3' % i)(x)
        x = Activation(activation='relu', name='Block_%s/act4' % i)(x)
        x = Dropout(rate=0.2, name='Block_%s/dr2' % i)(x)
        x = Conv2D(filters=output_size, kernel_size=16, strides=1, padding='same', kernel_initializer='he_uniform', name='Block_%s/conv5' % i)(x)

        x = Add()([shrt, x])

        prev_output_size = output_size

    x = BatchNormalization(name='bn4')(x)
    x = Activation(activation='relu', name='act5')(x)
    x = Flatten()(x)
    x = Dense(units=4, activation='softmax', name='dense')(x)

    return Model(inputs=a, outputs=x, name='stanford_ecg_net')

