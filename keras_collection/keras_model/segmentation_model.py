from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.regularizers import *

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


'''

original paper : large kernel matter
origianl paper url -> https://arxiv.org/abs/1703.02719

'''


def GCN(layer, k=18):
    conv1 = Conv2D(21, (k, 1), activation=None, padding='same')(layer)
    conv1 = Conv2D(21, (1, k), activation=None, padding='same')(conv1)

    conv2 = Conv2D(21, (1, k), activation=None, padding='same')(layer)
    conv2 = Conv2D(21, (k, 1), activation=None, padding='same')(conv2)

    merge = Add()([conv1, conv2])
    return merge


def BR(layer):
    conv = Conv2D(21, (3, 3), activation='relu', padding='same')(layer)
    conv = Conv2D(21, (3, 3), activation=None, padding='same')(conv)

    merge = Add()([layer, conv])
    return merge


def upsampling(layer):
    up = UpSampling2D(size=(2, 2))(layer)
    up = Conv2D(21, (3, 3), activation='relu', padding='same')(up)
    up = Conv2D(21, (3, 3), activation='relu', padding='same')(up)
    return up


def large_kernel_matters_model(img_h, img_w, classes, channel=3, summary=True):
    img_input = Input(shape=(img_h, img_w, channel), name="image_input")
    resnet = ResNet50(input_tensor=img_input, weights=None, include_top=False)

    res_5 = resnet.get_layer(name='activation_49').output
    res_4 = resnet.get_layer(name='activation_40').output
    res_3 = resnet.get_layer(name='activation_22').output
    res_2 = ZeroPadding2D(padding=((1, 0), (1, 0)))(resnet.get_layer(name='activation_10').output)

    gcn_1 = GCN(res_5)
    br_1 = BR(gcn_1)
    deconv_1 = upsampling(br_1)

    gcn_2 = GCN(res_4)
    br_2 = BR(gcn_2)
    merge_2 = Add()([deconv_1, br_2])
    br_2 = BR(merge_2)
    deconv_2 = upsampling(br_2)

    gcn_3 = GCN(res_3)
    br_3 = BR(gcn_3)
    merge_3 = Add()([deconv_2, br_3])
    br_3 = BR(merge_3)
    deconv_3 = upsampling(br_3)

    gcn_4 = GCN(res_2)
    br_4 = BR(gcn_4)
    merge_4 = Add()([deconv_3, br_4])
    br_4 = BR(merge_4)
    deconv_4 = upsampling(br_4)

    br_5 = BR(deconv_4)
    deconv_5 = upsampling(br_5)
    br_5 = BR(deconv_5)

    predict = Conv2D(classes, (1, 1), activation='sigmoid', padding='same')(br_5)

    model = Model(inputs=img_input, outputs=predict)

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = l2(0.0005)

    if summary:
        model.summary()

    return model