import numpy as np
import model as m
import tensorflow as tf
import time
from datetime import datetime
import os
import pandas as pd
from scipy.misc import imresize, imread
from tqdm import tqdm

class MultiGpu(object):
    def __init__(self,  image_size= 32, image_channel= 3, n_gpu=1, data_path= 'data.npz', n_class = 120,
                  max_step=100, batch_size=32, TOWER_NAME='tower'):

        self.n_gpu = n_gpu
        self.max_step = max_step
        self.batch_size = batch_size
        self.TOWER_NAME = TOWER_NAME
        self.image_size = image_size
        self.image_channel = image_channel
        self.data_path = data_path
        self.n_class = n_class
        self.NUM_EPOCHS_PER_DECAY = 30.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.16
        self.INITIAL_LEARNING_RATE = 0.01
        self.MOVING_AVERAGE_DECAY = 0.9
        self.RMSPROP_DECAY =0.9
        self.RMSPROP_MOMENTUM = 0.9
        self.RMSPROP_EPSILON = 1.0

    def load_test(self):

        df_test = pd.read_csv('sample_submission.csv')
        df = pd.read_csv('labels.csv')
        breed = set(df['breed'])
        class_to_num = dict(zip(breed, range(self.n_class)))
        NUMBER_TEST = len(df_test)
        X_test = np.zeros((NUMBER_TEST, self.image_size, self.image_size, 3), dtype=np.uint8)

        for i in tqdm(range(NUMBER_TEST)):
            X_test[i] = imresize(imread('test/%s.jpg' % df_test['id'][i]), (self.image_size, self.image_size))
        X_test = np.array(X_test, np.float32) / 255.

        return X_test , df_test, breed, class_to_num

    def load_data(self, path):

        load = np.load(path)
        # load data
        train_image = load['trainimg']
        train_label = load['trainlabel']

        number_train = train_image.shape[0]

        print('Number of train set : %s' % number_train)

        return train_image,train_label


    def tower_loss(self, scope, image, label):

        logits = m.inference(image)

        labels = tf.cast(label, tf.int64)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        _ = tf.add_n(tf.get_collection('losses'), name='total_loss')

        losses = tf.get_collection('losses', scope)

        total_loss = tf.add_n(losses, name='total_loss')

        return total_loss

    def average_gradients(self, tower_grads):

        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

                # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):

        train_image, train_label = self.load_data(self.data_path)

        n_train = len(train_image)

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            keep_prob = tf.placeholder(tf.float32)

            tower_grads = []

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.1), trainable=False)

            num_batches_per_epoch = (n_train / self.batch_size)
            decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

            lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            self.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            optimizer = tf.train.RMSPropOptimizer(lr, self.RMSPROP_DECAY,
                                            momentum=self.RMSPROP_MOMENTUM,
                                            epsilon=self.RMSPROP_EPSILON)

            randidx = np.random.randint(n_train, size=self.batch_size)
            batch_xs = train_image[randidx, :]
            batch_ys = train_label[randidx, :]

            batch_xs = tf.convert_to_tensor(batch_xs)
            batch_ys = tf.convert_to_tensor(batch_ys)

            batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                [batch_xs, batch_ys], capacity=2 * self.n_gpu)

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.n_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (self.TOWER_NAME, i)) as scope:
                            image_batch, label_batch = batch_queue.dequeue()

                            print('image_batch', image_batch)
                            print('image_label', label_batch)

                            loss = self.tower_loss(scope, image_batch,label_batch)
                            tf.get_variable_scope().reuse_variables()

                            grads = optimizer.compute_gradients(loss)
                            tower_grads.append(grads)

            grads = self.average_gradients(tower_grads)

            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))

            sess.run(init)

            tf.train.start_queue_runners(sess=sess)

            print('Session open.')
            learning_time = time.time()
            ckpt = tf.train.get_checkpoint_state('./checkpoint')

            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)

                else:
                    saver.restore(sess, ckpt.model_checkpoint_path)

            for step in range(self.max_step):

                loss_time = time.time()
                _, loss_value = sess.run([train_op, loss], feed_dict={keep_prob: 0.8})

                duration = time.time() - loss_time

                if step % 10 == 0 :
                    num_examples_per_step = self.batch_size * self.n_gpu
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / self.n_gpu

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')

                    print(format_str % (datetime.now(), step , loss_value, examples_per_sec,
                                        sec_per_batch))

                if step % 10000 == 0 or (step + 1) == self.max_step:
                    saver.save(sess, './checkpoint/model', global_step=step)

            finish_time = time.time() - learning_time
            print('Learning finish : ', finish_time)

            X_test, df_test, breed, class_to_num = self.load_test()
            tf.get_variable_scope().reuse_variables()
            logits = m.inference(X_test)
            softmax_logits = tf.nn.softmax(logits)
            predict = sess.run(softmax_logits)

            for b in breed:
                df_test[b] = predict[:, class_to_num[b]]

            df_test.to_csv('predict.csv', index=None)


MultiGpu().train()



