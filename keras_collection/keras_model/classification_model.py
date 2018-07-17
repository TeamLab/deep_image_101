from keras.models import *
from keras.layers import *
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3


def xception(img_h, img_w, channel, classes, gmp=True, gap=True, fc=False, summary=True):

    assert img_h < 200 or img_w < 200

    img_input = Input(shape=(img_h, img_w, channel), name="image_input")
    x = Xception(input_tensor=img_input, weights=None, include_top=False)

    # get latent featuremap
    latent = x.output

    # gap, gmp
    if gap:
        pool = GlobalAveragePooling2D()(latent)
    else:
        pool = GlobalMaxPooling2D()(latent)

    if fc:
        fc_layer = Dense(512, activation='relu')(pool)
        fc_layer = Dropout(0.2)(fc_layer)
        fc_layer = Dense(256, activation='relu')(fc_layer)
        fc_layer = Dropout(0.2)(fc_layer)
        predict = Dense(classes, activation='softmax')(fc_layer)
    else:
        predict = Dense(classes, activation='softmax')(pool)

    model = Model(inputs=img_input, outputs=predict)

    if summary:
        model.summary()

    return model


def inception_v3(img_h, img_w, channel, classes, gmp=True, gap=True, fc=False, summary=True):
    assert img_h < 200 or img_w < 200

    img_input = Input(shape=(img_h, img_w, channel), name="image_input")
    x = InceptionV3(input_tensor=img_input, weights=None, include_top=False)

    # get latent featuremap
    latent = x.output

    # gap, gmp
    if gap:
        pool = GlobalAveragePooling2D()(latent)
    else:
        pool = GlobalMaxPooling2D()(latent)

    if fc:
        fc_layer = Dense(512, activation='relu')(pool)
        fc_layer = Dropout(0.2)(fc_layer)
        fc_layer = Dense(256, activation='relu')(fc_layer)
        fc_layer = Dropout(0.2)(fc_layer)
        predict = Dense(classes, activation='softmax')(fc_layer)
    else:
        predict = Dense(classes, activation='softmax')(pool)

    model = Model(inputs=img_input, outputs=predict)

    if summary:
        model.summary()

    return model