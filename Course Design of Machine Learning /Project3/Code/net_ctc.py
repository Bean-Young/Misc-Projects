from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Reshape, Permute, Conv2D, Conv2DTranspose,
    ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten,
    concatenate, BatchNormalization, TimeDistributed
)
from tensorflow.keras.regularizers import l2


def dense_cnn(input, nclass):
    _dropout_rate = 0.4
    _weight_decay = 1e-4
    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.4, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter

input_layer = Input(shape=(32, 200, 1))  # 适配 Dense-CNN 结构的输入尺寸
n_classes = 10  

output_layer = dense_cnn(input_layer, n_classes)

model = Model(inputs=input_layer, outputs=output_layer)

model.summary()