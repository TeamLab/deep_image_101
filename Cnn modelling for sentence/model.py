import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import util


class ModellingCnn(object):
    def __init__(self, batch_size=64, embedding_size=100, first_filter=7, second_filter=5, top_k=4, total_layer=3,
                 first_featuremap=8, second_featuremap =5, training_epochs=30):
        self.batch_size = batch_size
        self.sequence_length = None
        self.embedding_size = embedding_size
        self.num_class = None
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.first_filter = first_filter
        self.second_filter = second_filter
        self.top_k = top_k
        self.total_layer = total_layer
        self.first_featuremap = first_featuremap
        self.second_featuremap = second_featuremap
        self.training_epochs = training_epochs
        self.initial_lr = 0.1
        self.step_size = 20000
        self.decay_ratio = 0.005

    def fully_connected(self, input):
        fc_1 = slim.fully_connected(input, 256, activation_fn=tf.nn.relu)
        fc_1 = tf.nn.dropout(fc_1, self.keep_prob)
        fc_2 = slim.fully_connected(fc_1, 256, activation_fn=tf.nn.relu)
        fc_2 = tf.nn.dropout(fc_2, self.keep_prob)
        output = slim.fully_connected(fc_2, self.num_class, activation_fn=None)

        return output

    def k_max_pool(self, layer, current_layer):

        sequence_length = int(np.shape(layer)[1])
        layer = tf.reshape(layer, [-1, sequence_length])
        k = max(self.top_k, int(((self.total_layer - current_layer) / self.total_layer) * sequence_length))
        k_max_value, _ = tf.nn.top_k(layer, k)
        k_max_value = tf.expand_dims(k_max_value, axis=-1)
        k_max_value = tf.expand_dims(k_max_value, axis=-1)

        return k_max_value

    def folding(self, layer):

        number_split = (int(np.shape(layer)[1]))/2

        folding = []
        split_value = tf.split(layer, num_or_size_splits=int(number_split), axis=1)
        for i in split_value:
            sum_value = tf.reduce_sum(i, axis=1)
            sum_value = tf.reshape(sum_value, [-1, 1])

            folding.append(sum_value)

        output = tf.concat(folding, axis=1)

        return output

    def model(self):

        util.load_data(self)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.input_y = tf.placeholder(tf.float32, [None, self.num_class])
        self.keep_prob = tf.placeholder(tf.float32)

        # word embedding
        embedding_w = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
            name='embed_w')
        embedded_chars = tf.nn.embedding_lookup(embedding_w, self.input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        w_initialize = tf.contrib.layers.xavier_initializer()
        b_initialize = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

        # first one dimensional wide-convolution layer ( zero padding )
        feature_map = []
        for i in range(self.first_featuremap):
            with tf.variable_scope('first_convolution_layer'+str(i)):
                conv_w = tf.get_variable('conv1',
                                         shape=[self.first_filter, self.embedding_size, 1, 1],
                                         dtype=tf.float32,
                                         regularizer=regularizer,
                                         initializer=w_initialize)

                conv_b = tf.get_variable('bias1',
                                         shape=[1, ],
                                         dtype=tf.float32,
                                         regularizer=regularizer,
                                         initializer=b_initialize)

                layer_1 = tf.nn.conv2d(embedded_chars_expanded, conv_w,
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')

                layer_1 = tf.nn.bias_add(layer_1, conv_b)
                layer_1 = tf.nn.relu(layer_1)
                layer_1 = tf.nn.dropout(layer_1, self.keep_prob)

                pool_layer = self.k_max_pool(layer_1, 1)

            feature_map.append(pool_layer)

        logit = []
        for i, x in enumerate(feature_map):

            concat_layer = []
            for j in range(self.second_featuremap):
                with tf.variable_scope('second_convolution_layer'+ str(i) +str(j)):
                    conv_w_2 = tf.get_variable('conv2',
                                               shape=[self.second_filter, 1, 1, 1],
                                               dtype=tf.float32,
                                               regularizer=regularizer,
                                               initializer=w_initialize)

                    conv_b_2 = tf.get_variable('bias2',
                                               shape=[1, ],
                                               dtype=tf.float32,
                                               regularizer=regularizer,
                                               initializer=b_initialize)

                    layer_2 = tf.nn.conv2d(x, conv_w_2,
                                           strides=[1, 1, 1, 1],
                                           padding='VALID')

                    # 16, 1, 1, 1
                    layer_2 = tf.nn.bias_add(layer_2, conv_b_2)
                    layer_2 = tf.nn.relu(layer_2)
                    layer_2 = tf.nn.dropout(layer_2, self.keep_prob)

                    folding_layer = self.folding(layer_2)

                    pool_layer_2 = self.k_max_pool(folding_layer, 2)

                    concat_layer.append(pool_layer_2)
            sub_layer = tf.concat(concat_layer, 1)
            logit.append(sub_layer)
        conv_layer_output = tf.concat(logit, 1)
        conv_layer_output = tf.reshape(conv_layer_output, [-1, int(np.shape(conv_layer_output)[1])])

        # fully-connected layer
        logits = self.fully_connected(conv_layer_output)

        print('Logit shape is ', np.shape(logits))

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

        with tf.name_scope('accuracy'):
            predict = tf.argmax(logits, 1)
            correct_prediction = tf.equal(tf.argmax(self.input_y, 1), predict)
            correct_prediction = tf.cast(correct_prediction, tf.float32)

            self.accuracy = tf.reduce_mean(correct_prediction)

        return loss


    def train(self):

        loss = self.model()

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.initial_lr, global_step,
                                                   self.step_size, self.decay_ratio,
                                                   staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)

        train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        index = 0
        pre_acc = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print('\nStart training...')

            for epoch in range(int(self.training_epochs)):
                avg_cost = 0
                total_batch = int(len(self.x_train) / self.batch_size)

                for i in range(total_batch):
                    if (index + 1) * self.batch_size > self.x_train.shape[0]:
                        _, loss_ = sess.run([train_op, loss], feed_dict={
                            self.input_x: self.x_train[index * self.batch_size::],
                            self.input_y: self.y_train[index * self.batch_size::],
                            self.keep_prob: 1.0})
                        index = 0
                    else:
                        _, loss_ = sess.run([train_op, loss], feed_dict={
                            self.input_x: self.x_train[index * self.batch_size:(index + 1) * self.batch_size],
                            self.input_y: self.y_train[index * self.batch_size:(index + 1) * self.batch_size],
                            self.keep_prob: 1.0})
                        index += 1

                    avg_cost += loss_ / total_batch

                if epoch % 1 == 0:
                    # FIXME
                    test_acc = util.calc_acc(self, sess, self.x_test, self.y_test)
                    if test_acc >= pre_acc:
                        print('Epoch: %s, Cost: %s,Test accuracy: %s' % (epoch + 1, avg_cost, test_acc))