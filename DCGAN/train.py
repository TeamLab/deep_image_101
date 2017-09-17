#! /usr/bin/python
# -*- coding: utf8 -*-

import time, os, re, nltk
from utils import *
from model import *
import model

print("Loading data from pickle ...")
import pickle
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)

images_train = np.array(images_train)
images_test = np.array(images_test)

ni = int(np.ceil(np.sqrt(batch_size)))

tl.files.exists_or_mkdir("result")
tl.files.exists_or_mkdir("checkpoint")
save_dir = "checkpoint"

print('batch size ', batch_size)
def main_train():

    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')


    ## training inference for txt2img
    generator_txt2img = model.generator
    discriminator_txt2img = model.discriminator

    net_rnn = rnn_embed(t_real_caption, is_train=True, reuse=False)
    net_fake_image, _ = generator_txt2img(t_z,
                    net_rnn.outputs,
                    is_train=True, reuse=False, batch_size=batch_size)

    net_d, disc_fake_image_logits = discriminator_txt2img(
                    net_fake_image, net_rnn.outputs, is_train=True, reuse=False)
    _, disc_real_image_logits = discriminator_txt2img(
                    t_real_image, net_rnn.outputs, is_train=True, reuse=True)
    _, disc_mismatch_logits = discriminator_txt2img(
                    t_real_image,
                    rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs,
                    is_train=True, reuse=True)

    ## testing inference for txt2img
    net_g, _ = generator_txt2img(t_z,
                    rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                    is_train=False, reuse=True, batch_size=batch_size)

    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
    g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')

    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.0002
    lr_decay = 0.5      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    tl.layers.initialize_global_variables(sess)

    # seed for generation, z and sentence ids
    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

    sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
                      ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
                      ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
                      ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
                      ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
                      ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
                      ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
                      ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)

    # sample_sentence = captions_ids_test[0:sample_size]
    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID

    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    n_epoch = 600
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)

    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            # get matched text
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_real_caption = captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
            # get real image
            b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]

            # get wrong caption
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_wrong_caption = captions_ids_train[idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')

            # get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
            # [0, 255] --> [-1, 1] + augmentation
            b_real_images = threading_data(b_real_images, prepro_img, mode='train')

            # updates D
            D_loss, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image : b_real_images,
                            t_wrong_caption : b_wrong_caption,
                            t_real_caption : b_real_caption,
                            t_z : b_z})
            # updates G
            G_loss, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_caption : b_real_caption,
                            t_z : b_z})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, D_loss, G_loss))

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([net_g, net_rnn.outputs], feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_z : sample_seed})

            save_images(img_gen, [ni, ni], 'result/train_{:02d}.png'.format(epoch))

        # save model
        if (epoch != 0) and (epoch % 10) == 0:

            saver.save(sess, './checkpoint/model-%s'%epoch)
            print("model-%s saved." %epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                       help='train, train_encoder, translation')

    args = parser.parse_args()

    if args.mode == "train":
        main_train()
