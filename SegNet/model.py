from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.vgg16 import VGG16

import keras.backend as K


def zeropadding_layer(inputs):

    layer_height = K.shape(inputs)[1]
    layer_width = K.shape(inputs)[2]

    padding_w = int(layer_height % 2)
    padding_h = int(layer_width % 2)

    return ZeroPadding2D(((padding_h, 0), (0, padding_w)))(inputs)


def binary_dice_coefficient(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = y_pred_f > 0.5
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def binary_dice_loss(y_true, y_pred):
    return 1. - binary_dice_coefficient(y_true, y_pred)


def multi_class_dice_coefficient(y_true, y_pred, smooth=1e-7):

    num_class = K.shape(y_pred)[-1]

    # Ignores background pixel label 0
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_class)[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def multi_class_dice_loss(y_true, y_pred):
    # Dice loss to minimize.

    return 1. - multi_class_dice_coefficient(y_true, y_pred)


def SegNet(img_shape, num_classes, use_bias=True, multi_class=True, summary=True):

    input_img = Input(shape=img_shape)

    '''
    SegNet : encoder architecture consist of 13 layers of VGG architecture.
    decoder up-convolutional layers shape is difference when encoder layer has odd dimension
    so we use zero-padding because up-sampling layer have to same dimension correspond pooling indices.
    
    '''
    # encoder

    # encoder block 1
    conv_1 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(input_img)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_1)
    conv_1 = Activation('relu')(conv_1)
    conv_1 = zeropadding_layer(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    # encoder block 2
    conv_2 = Conv2D(128, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(pool_1)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = Conv2D(128, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_2)
    conv_2 = Activation('relu')(conv_2)
    conv_2 = zeropadding_layer(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    # encoder block 3
    conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(pool_2)
    conv_3 = Activation('relu')(conv_3)
    conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_3)
    conv_3 = Activation('relu')(conv_3)
    conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_3)
    conv_3 = Activation('relu')(conv_3)
    conv_3 = zeropadding_layer(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    # encoder block 4
    conv_4 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(pool_3)
    conv_4 = Activation('relu')(conv_4)
    conv_4 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_4)
    conv_4 = Activation('relu')(conv_4)
    conv_4 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_4)
    conv_4 = Activation('relu')(conv_4)
    conv_4 = zeropadding_layer(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    # encoder block 5
    conv_5 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(pool_4)
    conv_5 = Activation('relu')(conv_5)
    conv_5 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_5)
    conv_5 = Activation('relu')(conv_5)
    conv_5 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(conv_5)
    conv_5 = Activation('relu')(conv_5)
    conv_5 = zeropadding_layer(conv_5)
    pool_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)

    # decoder

    # decoder block 1
    up_conv_1 = UpSampling2D(size=(2, 2))(pool_5)
    up_conv_1 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_1)
    up_conv_1 = Activation('relu')(up_conv_1)
    up_conv_1 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_1)
    up_conv_1 = Activation('relu')(up_conv_1)
    up_conv_1 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_1)
    up_conv_1 = Activation('relu')(up_conv_1)

    # decoder block 2
    up_conv_2 = UpSampling2D(size=(2, 2))(up_conv_1)
    up_conv_2 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_2)
    up_conv_2 = Activation('relu')(up_conv_2)
    up_conv_2 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_2)
    up_conv_2 = Activation('relu')(up_conv_2)
    up_conv_2 = Conv2D(512, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_2)
    up_conv_2 = Activation('relu')(up_conv_2)

    # decoder block 3
    up_conv_3 = UpSampling2D(size=(2, 2))(up_conv_2)
    up_conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_3)
    up_conv_3 = Activation('relu')(up_conv_3)
    up_conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_3)
    up_conv_3 = Activation('relu')(up_conv_3)
    up_conv_3 = Conv2D(256, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_3)
    up_conv_3 = Activation('relu')(up_conv_3)

    # decoder block 4
    up_conv_4 = UpSampling2D(size=(2, 2))(up_conv_3)
    up_conv_4 = Conv2D(128, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_4)
    up_conv_4 = Activation('relu')(up_conv_4)
    up_conv_4 = Conv2D(128, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_4)
    up_conv_4 = Activation('relu')(up_conv_4)

    # decoder block 5
    up_conv_5 = UpSampling2D(size=(2, 2))(up_conv_4)
    up_conv_5 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_5)
    up_conv_5 = Activation('relu')(up_conv_5)
    up_conv_5 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_5)
    up_conv_5 = Activation('relu')(up_conv_5)

    # predict
    predict = Conv2D(num_classes, (1, 1), strides=(1, 1), activation=None, use_bias=use_bias, padding='same')(up_conv_5)

    model = Model(input_img, predict)

    if summary:
        model.summary()

    if multi_class:
        model.compile(loss=[multi_class_dice_loss], metrics=[multi_class_dice_coefficient],
                      optimizer=Adam(lr=0.001))
    else:
        model.compile(loss=[binary_dice_loss], metrics=[binary_dice_coefficient],
                      optimizer=Adam(lr=0.001))

    return model






