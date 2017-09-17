#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import ops

batch_size = 64
z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb

t_dim = 128         # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0

def rnn_embed(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')

        return network


def generator(input_z, t_txt=None, is_train=True, reuse=False, batch_size=batch_size):

    g_bn0 = ops.batch_norm(name='g_bn0')
    g_bn1 = ops.batch_norm(name='g_bn1')
    g_bn2 = ops.batch_norm(name='g_bn2')
    g_bn3 = ops.batch_norm(name='g_bn3')

    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        z_concat = tf.concat([input_z, t_txt], 1)
        z_ = ops.linear(z_concat, gf_dim * 8 * s16 * s16, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s16, s16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0))
        h1 = ops.deconv2d(h0, [batch_size, s8, s8, gf_dim * 4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1))

        h2 = ops.deconv2d(h1, [batch_size, s4, s4, gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = ops.deconv2d(h2, [batch_size, s2, s2, gf_dim* 1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = ops.deconv2d(h3, [batch_size, s, s, 3], name='g_h4')

    return h4, tf.tanh(h4)

def discriminator(input_images, t_txt=None, is_train=True, reuse=False):
    d_bn1 = ops.batch_norm(name='d_bn1')
    d_bn2 = ops.batch_norm(name='d_bn2')
    d_bn3 = ops.batch_norm(name='d_bn3')
    d_bn4 = ops.batch_norm(name='d_bn4')

    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        h0 = ops.lrelu(ops.conv2d(input_images, df_dim, name='d_h0_conv'))  # 32
        h1 = ops.lrelu(d_bn1(ops.conv2d(h0, df_dim * 2, name='d_h1_conv')))  # 16
        h2 = ops.lrelu(d_bn2(ops.conv2d(h1, df_dim * 4, name='d_h2_conv')))  # 8
        h3 = ops.lrelu(d_bn3(ops.conv2d(h2, df_dim * 8, name='d_h3_conv')))  # 4

        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_txt, t_dim, 'd_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
        tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings')

        h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')
        h3_new = ops.lrelu(
            d_bn4(ops.conv2d(h3_concat, df_dim * 8, 1, 1, 1, 1, name='d_h3_conv_new')))  # 4

        h4 = ops.linear(tf.reshape(h3_new, [batch_size, -1]), 1, 'd_h3_lin')

    return tf.nn.sigmoid(h4), h4
