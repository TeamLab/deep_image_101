from keras.models import *
from keras.layers import *
from keras.applications.xception import Xception


def baseline_regression(img_h, img_w, channel, gmp=True, gap=True, summary=True):
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

    fc = Dense(512, activation=None)(pool)
    fc = Dropout(0.2)(fc)
    fc = Dense(1, activation=None)(fc)

    model = Model(inputs=img_input, outputs=fc)

    if summary:
        model.summary()

    return model