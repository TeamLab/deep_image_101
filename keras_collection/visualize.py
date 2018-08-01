from skimage import io, exposure
import matplotlib.payplot as plt
import keras.backend as K
import numpy as np
import cv2


def cam(model, img_dir, out_dir, img_normalize=True, histogram_equalizer=True):

    origin_img = io.imread(img_dir)

    img_h = origin_img.shape[0]
    img_w = origin_img.shape[1]

    assert origin_img.shape[0] == img_h and origin_img.shape[1] == img_w

    # inference image
    if histogram_equalizer:
        origin_img = exposure.equalize_hist(origin_img)
    if img_normalize:
        origin_img = (origin_img - np.mean(origin_img)) / np.std(origin_img)

    origin_img = np.reshape(origin_img, [1, img_h, img_w, 1])

    score = model.predict(origin_img)
    predict = np.argmax(score, axis=1)
    select_score = score[0][predict[0]]

    print("Model predict label : {}, score : {:.5f}".format(predict[0], select_score))

    # get weight after GAP
    weights = model.layers[-1].get_weights()[0]
    biases = model.layers[-1].get_weights()[1]

    predict_weight = np.reshape(weights[:, predict], [-1])
    predict_bias = biases[predict]

    # get layers before GAP
    get_latent_conv = K.function([model.layers[0].input],
                                 [model.layers[-3].output])
    layer_output = get_latent_conv([origin_img])[0]

    # cal cam
    activation_map = (layer_output * predict_weight)
    activation_map = np.sum(activation_map, axis=-1)

    if activation_map.shape[0] == 1:
        activation_map = np.squeeze(activation_map, axis=0)

    # hitmap
    activation_map /= np.max(activation_map)
    activation_map = cv2.resize(activation_map, (img_w, img_h))
    up_cam = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    up_cam[np.where(activation_map < 0.6)] = 0

    # origin image convert uint8
    origin_img = origin_img * 255
    origin_img = origin_img.astype(np.uint8)
    if len(origin_img.shape) != 3:
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2RGB)

    # coordinate scale
    img = (up_cam * 0.3) + (origin_img * 0.5)
    result = img.astype(np.uint8)
    plt.imshow(result)
    plt.imsave(out_dir, result)
    print("save image at {}".format(out_dir))


'''
Grad-Cam : https://arxiv.org/abs/1610.02391
class activation map should use GAP, so model structure is limited.
But grad-cam uses gradient. It is not affected by the model structure.
'''

def grad_cam(model, img_tensor, save_dir):

    img_h = img_tensor.shape[0]
    img_w = img_tensor.shape[1]
    img_c = img_tensor.shape[2]

    reshape_img = np.reshape(img_tensor, [1, img_h, img_w, img_c])
    y_pred = model.predict(reshape_img)
    y_pred = np.argmax(y_pred, axis=1)
    print("predict label : {}".format(y_pred))

    inputs = model.input
    y_c = model.output.op.inputs[0][0, y_pred]
    A_k = []

    for i, layer in enumerate(model.layers):
        if model.layers[i].name == 'latent_conv':
            A_k = model.get_layer(name='latent_conv').output

    a_k = K.function([inputs], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [target_layer_output, grad_val, model_output] = a_k([img_tensor])

    target_layer_output = target_layer_output[0]
    grad_val = grad_val[0]
    grad_weights = np.mean(grad_val, axis=(0, 1))

    grad_cam = np.zeros(dtype=np.float32, shape=target_layer_output.shape[0:2])
    for k, w in enumerate(grad_weights):
        grad_cam += w * target_layer_output[:, :, k]

    grad_cam = cv2.resize(grad_cam, (img_h, img_w))
    grad_cam = np.maximum(grad_cam, 0)
    activationmap = grad_cam / grad_cam.max()

    # heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * activationmap), cv2.COLORMAP_JET)

    # convert grayscale img to BGR img ( cv2 default BGR not RBG )
    if img_c != 3:
        img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_GRAY2BGR)

    activation_img = (heatmap * 0.3) + (img_tensor * 0.7)
    cv2.imwrite(save_dir, activation_img)