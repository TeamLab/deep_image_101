from keras.models import Model
from keras.layers import Dense, Convolution2D
import numpy as np


def svd_orthonormal(shape):
    # Orthonorm init code is taked from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_activations(model, layer, x_batch):
    intermediate_layer_model = Model(
        inputs=model.get_input_at(0),
        outputs=layer.get_output_at(0)
    )
    activations = intermediate_layer_model.predict(x_batch)
    return activations


def LSUV(model, batch_x, TOL_var=0.1, T_max=5):
    Layerlist = (Dense, Convolution2D)
    layers_inintialized = 0
    for layer in model.layers:

        if not isinstance(layer, Layerlist):
            continue

        layers_inintialized += 1
        weights_and_biases = layer.get_weights()
        W_l = weights_and_biases[0]
        biases = weights_and_biases[1]
        W_l = svd_orthonormal(W_l.shape)
        layer.set_weights([W_l, biases])
        forward = get_activations(model, layer, batch_x)

        B_var = np.var(forward)

        for T_i in range(T_max):

            if abs(B_var - 1.0) > TOL_var:
                break
            if np.abs(np.sqrt(B_var)) < 1e-7:
                break

            weights_and_biases = layer.get_weights()
            W_l = weights_and_biases[0]
            biases = weights_and_biases[1]
            W_l /= np.sqrt(B_var)
            layer.set_weights([W_l, biases])

            forward = get_activations(model, layer, batch_x)
            B_var = np.var(forward)

    return model