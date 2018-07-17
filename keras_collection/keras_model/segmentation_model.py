from keras.models import *
from keras.layers import *


def unet(img_h, img_w, channel, classes=1, num_filter=64, kernel_size=3, init='he_normal'):
    inputs = Input((img_h, img_w, channel))
    conv1 = Conv2D(num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(inputs)
    conv1 = Conv2D(num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(pool1)
    conv2 = Conv2D(2*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(pool2)
    conv3 = Conv2D(4*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(6*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(pool3)
    conv4 = Conv2D(6*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(8*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(pool4)
    conv5 = Conv2D(8*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(6*num_filter, 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer=init)(UpSampling2D(size=(2, 2))(drop5))

    merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(6*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(merge6)
    conv6 = Conv2D(6*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv6)

    up7 = Conv2D(4*num_filter, 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv6))

    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(4*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(merge7)
    conv7 = Conv2D(4*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv7)

    up8 = Conv2D(2*num_filter, 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv7))

    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(2*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(merge8)
    conv8 = Conv2D(2*num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv8)

    up9 = Conv2D(num_filter, 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer=init)(UpSampling2D(size=(2, 2))(conv8))

    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(merge9)
    conv9 = Conv2D(num_filter, kernel_size, activation='relu', padding='same', kernel_initializer=init)(conv9)

    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    return model