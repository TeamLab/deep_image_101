import tensorflow as tf


def lrelu(x):
  return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)


def model(image_shape,keep_prob):

    with tf.variable_scope('stem') as scope:
        conv1_weight = tf.get_variable('conv1_weight', [3, 3, 3, 32],initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(image_shape, conv1_weight, strides=[1, 2, 2, 1], padding='VALID')
        conv1 = lrelu(conv1)
        conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9997, epsilon=0.001)
        print(conv1)

        conv2_weight = tf.get_variable('conv2_weight', [3, 3, 32, 32],initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(conv1, conv2_weight, strides=[1, 1, 1, 1], padding='VALID')
        conv2 = lrelu(conv2)
        conv2 = tf.contrib.layers.batch_norm(conv2, decay=0.9997, epsilon=0.001)
        print(conv2)

        conv3_weight = tf.get_variable('conv3_weight', [3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.nn.conv2d(conv2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = lrelu(conv3)
        conv3 = tf.contrib.layers.batch_norm(conv3, decay=0.9997, epsilon=0.001)
        print(conv3)

        # split layers
        tower1_weight = tf.get_variable('tower1_weight', [3, 3, 64, 96],initializer=tf.contrib.layers.xavier_initializer())
        tower1 = tf.nn.conv2d(conv3, tower1_weight, strides=[1, 2, 2, 1], padding='VALID')
        tower1 = lrelu(tower1)
        tower1 = tf.contrib.layers.batch_norm(tower1, decay=0.9997, epsilon=0.001)
        print(tower1)

        tower_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(tower_pool)

        concat_1 = tf.concat([tower1,tower_pool],axis=3)
        print(concat_1)

        conv4_weight = tf.get_variable('conv4_weight', [1, 1, 160, 64],initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.nn.conv2d(concat_1, conv4_weight, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = lrelu(conv4)
        conv4 = tf.contrib.layers.batch_norm(conv4, decay=0.9997, epsilon=0.001)

        conv5_weight = tf.get_variable('conv5_weight', [7, 1, 64, 64],initializer=tf.contrib.layers.xavier_initializer())
        conv5 = tf.nn.conv2d(conv4, conv5_weight, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = lrelu(conv5)
        conv5 = tf.contrib.layers.batch_norm(conv5, decay=0.9997, epsilon=0.001)

        conv6_weight = tf.get_variable('conv6_weight', [1, 7, 64, 64],initializer=tf.contrib.layers.xavier_initializer())
        conv6 = tf.nn.conv2d(conv5, conv6_weight, strides=[1, 1, 1, 1], padding='SAME')
        conv6 = lrelu(conv6)
        conv6 = tf.contrib.layers.batch_norm(conv6, decay=0.9997, epsilon=0.001)

        conv7_weight = tf.get_variable('conv7_weight', [3, 3, 64, 96],initializer=tf.contrib.layers.xavier_initializer())
        conv7 = tf.nn.conv2d(conv6, conv7_weight, strides=[1, 1, 1, 1], padding='VALID')
        conv7 = lrelu(conv7)
        conv7 = tf.contrib.layers.batch_norm(conv7, decay=0.9997, epsilon=0.001)
        print(conv7)

        conv8_weight = tf.get_variable('conv8_weight', [1, 1, 160, 64],initializer=tf.contrib.layers.xavier_initializer())
        conv8 = tf.nn.conv2d(concat_1, conv8_weight, strides=[1, 1, 1, 1], padding='SAME')
        conv8 = lrelu(conv8)
        conv8 = tf.contrib.layers.batch_norm(conv8, decay=0.9997, epsilon=0.001)

        conv9_weight = tf.get_variable('conv9_weight', [3, 3, 64, 96],initializer=tf.contrib.layers.xavier_initializer())
        conv9 = tf.nn.conv2d(conv8, conv9_weight, strides=[1, 1, 1, 1], padding='VALID')
        conv9 = lrelu(conv9)
        conv9 = tf.contrib.layers.batch_norm(conv9, decay=0.9997, epsilon=0.001)

        concat_2 = tf.concat([conv7,conv9],3)
        print(concat_2)

        pool2 = tf.nn.max_pool(concat_2, ksize=[1,3,3,1], strides=[1, 2, 2, 1], padding='VALID')
        print(pool2)

        conv10_weight = tf.get_variable('conv10_weight', [3, 3, 192, 192],initializer=tf.contrib.layers.xavier_initializer())
        conv10 = tf.nn.conv2d(concat_2, conv10_weight, strides=[1, 2, 2, 1], padding='VALID')
        conv10 = lrelu(conv10)
        conv10 = tf.contrib.layers.batch_norm(conv10, decay=0.9997, epsilon=0.001)
        print(conv10)

        # stem size is 35 X 35 X 384
        stem = tf.concat([conv10, pool2], 3, name=scope.name)

    # Inception-A start
    with tf.variable_scope('Inception-A') as scope:
        inception_a_pool1 = tf.nn.avg_pool(stem,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
        inception_a_1_weight = tf.get_variable('inception_a_1_weight', [1, 1, 384, 96],initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv1 = tf.nn.conv2d(inception_a_pool1, inception_a_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv1 = lrelu(inception_a_conv1)
        inception_a_conv1 = tf.contrib.layers.batch_norm(inception_a_conv1, decay=0.9997, epsilon=0.001)
        print(inception_a_conv1)

        inception_a_2_weight = tf.get_variable('inception_a_2_weight', [1, 1, 384, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv2 = tf.nn.conv2d(stem, inception_a_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv2 = lrelu(inception_a_conv2)
        inception_a_conv2 = tf.contrib.layers.batch_norm(inception_a_conv2, decay=0.9997, epsilon=0.001)
        print(inception_a_conv2)

        inception_a_3_weight = tf.get_variable('inception_a_3_weight', [1, 1, 384, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv3 = tf.nn.conv2d(stem, inception_a_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv3 = lrelu(inception_a_conv3)
        inception_a_conv3 = tf.contrib.layers.batch_norm(inception_a_conv3, decay=0.9997, epsilon=0.001)
        print(inception_a_conv3)

        inception_a_3_1_weight = tf.get_variable('inception_a_3_1_weight', [3, 3, 96, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv3 = tf.nn.conv2d(inception_a_conv3, inception_a_3_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv3 = lrelu(inception_a_conv3)
        inception_a_conv3 = tf.contrib.layers.batch_norm(inception_a_conv3, decay=0.9997, epsilon=0.001)
        print(inception_a_conv3)


        inception_a_4_weight = tf.get_variable('inception_a_4_weight', [1, 1, 384, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv4 = tf.nn.conv2d(stem, inception_a_4_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv4 = lrelu(inception_a_conv4)
        inception_a_conv4 = tf.contrib.layers.batch_norm(inception_a_conv4, decay=0.9997, epsilon=0.001)
        print(inception_a_conv4)

        inception_a_4_1_weight = tf.get_variable('inception_a_4_1_weight', [3, 3, 96, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv4 = tf.nn.conv2d(inception_a_conv4, inception_a_4_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv4 = lrelu(inception_a_conv4)
        inception_a_conv4 = tf.contrib.layers.batch_norm(inception_a_conv4, decay=0.9997, epsilon=0.001)
        print(inception_a_conv4)

        inception_a_4_2_weight = tf.get_variable('inception_a_4_2_weight', [3, 3, 96, 96],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_a_conv4 = tf.nn.conv2d(inception_a_conv4, inception_a_4_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_a_conv4 = lrelu(inception_a_conv4)
        inception_a_conv4 = tf.contrib.layers.batch_norm(inception_a_conv4, decay=0.9997, epsilon=0.001)
        print(inception_a_conv4)

        inception_A = tf.concat([inception_a_conv1,inception_a_conv2,inception_a_conv3,
                                 inception_a_conv4],3,name=scope.name)
        print(inception_A)

    # Reduction-A start!
    with tf.variable_scope('Reduction-A') as scope:

        reduction_a_1 = tf.nn.max_pool(inception_A,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

        reduction_a_2_weight = tf.get_variable('reduction_a_2_weight', [3, 3, 384, 384],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        reduction_a_2 = tf.nn.conv2d(inception_A, reduction_a_2_weight, strides=[1, 2, 2, 1], padding='VALID')
        reduction_a_2 = lrelu(reduction_a_2)
        reduction_a_2 = tf.contrib.layers.batch_norm(reduction_a_2, decay=0.9997, epsilon=0.001)
        print(reduction_a_2)

        reduction_a_3_weight = tf.get_variable('reduction_a_3_weight', [1, 1, 384, 192],
                                               initializer=tf.contrib.layers.xavier_initializer())
        reduction_a_3 = tf.nn.conv2d(inception_A, reduction_a_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_a_3 = lrelu(reduction_a_3)
        reduction_a_3 = tf.contrib.layers.batch_norm(reduction_a_3, decay=0.9997, epsilon=0.001)
        print(reduction_a_3)

        reduction_a_3_1_weight = tf.get_variable('reduction_a_3_1_weight', [3, 3, 192, 224],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        reduction_a_3 = tf.nn.conv2d(reduction_a_3, reduction_a_3_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_a_3 = lrelu(reduction_a_3)
        reduction_a_3 = tf.contrib.layers.batch_norm(reduction_a_3, decay=0.9997, epsilon=0.001)
        print(reduction_a_3)

        reduction_a_3_2_weight = tf.get_variable('reduction_a_3_2_weight', [3, 3, 224, 256],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        reduction_a_3 = tf.nn.conv2d(reduction_a_3, reduction_a_3_2_weight, strides=[1, 2, 2, 1], padding='VALID')
        reduction_a_3 = lrelu(reduction_a_3)
        reduction_a_3 = tf.contrib.layers.batch_norm(reduction_a_3, decay=0.9997, epsilon=0.001)
        print(reduction_a_3)

        Reduction_A = tf.concat([reduction_a_1,reduction_a_2,reduction_a_3],3,name=scope.name)
        print(Reduction_A)

    #Inception-B start!
    with tf.variable_scope('Inception-B') as scope:

        inception_b_pool1 = tf.nn.avg_pool(Reduction_A, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        inception_b_1_weight = tf.get_variable('inception_b_1_weight', [1, 1, 1024, 128],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv1 = tf.nn.conv2d(inception_b_pool1, inception_b_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv1 = lrelu(inception_b_conv1)
        inception_b_conv1 = tf.contrib.layers.batch_norm(inception_b_conv1, decay=0.9997, epsilon=0.001)
        print(inception_b_conv1)


        inception_b_2_weight = tf.get_variable('inception_b_2_weight', [1, 1, 1024, 384],
                                               initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv2 = tf.nn.conv2d(Reduction_A, inception_b_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv2 = lrelu(inception_b_conv2)
        inception_b_conv2 = tf.contrib.layers.batch_norm(inception_b_conv2, decay=0.9997, epsilon=0.001)
        print(inception_b_conv2)

        inception_b_3_weight = tf.get_variable('inception_b_3_weight', [1, 1, 1024, 192],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv3 = tf.nn.conv2d(Reduction_A, inception_b_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv3 = lrelu(inception_b_conv3)
        inception_b_conv3 = tf.contrib.layers.batch_norm(inception_b_conv3, decay=0.9997, epsilon=0.001)

        inception_b_3_1_weight = tf.get_variable('inception_b_3_1_weight', [1, 7, 192, 224],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv3 = tf.nn.conv2d(inception_b_conv3, inception_b_3_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv3 = lrelu(inception_b_conv3)
        inception_b_conv3 = tf.contrib.layers.batch_norm(inception_b_conv3, decay=0.9997, epsilon=0.001)

        inception_b_3_2_weight = tf.get_variable('inception_b_3_2_weight', [1, 7, 224, 256],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv3 = tf.nn.conv2d(inception_b_conv3, inception_b_3_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv3 = lrelu(inception_b_conv3)
        inception_b_conv3 = tf.contrib.layers.batch_norm(inception_b_conv3, decay=0.9997, epsilon=0.001)


        inception_b_4_weight = tf.get_variable('inception_b_4_weight', [1, 1, 1024, 192],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv4 = tf.nn.conv2d(Reduction_A, inception_b_4_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv4 = lrelu(inception_b_conv4)
        inception_b_conv4 = tf.contrib.layers.batch_norm(inception_b_conv4, decay=0.9997, epsilon=0.001)

        inception_b_4_1_weight = tf.get_variable('inception_b_4_1_weight', [1, 7, 192, 192],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv4 = tf.nn.conv2d(inception_b_conv4, inception_b_4_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv4 = lrelu(inception_b_conv4)
        inception_b_conv4 = tf.contrib.layers.batch_norm(inception_b_conv4, decay=0.9997, epsilon=0.001)

        inception_b_4_2_weight = tf.get_variable('inception_b_4_2_weight', [7, 1, 192, 224],initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv4 = tf.nn.conv2d(inception_b_conv4, inception_b_4_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv4 = lrelu(inception_b_conv4)
        inception_b_conv4 = tf.contrib.layers.batch_norm(inception_b_conv4, decay=0.9997, epsilon=0.001)

        inception_b_4_3_weight = tf.get_variable('inception_b_4_3_weight', [1, 7, 224, 224],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv4 = tf.nn.conv2d(inception_b_conv4, inception_b_4_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv4 = lrelu(inception_b_conv4)
        inception_b_conv4 = tf.contrib.layers.batch_norm(inception_b_conv4, decay=0.9997, epsilon=0.001)

        inception_b_4_4_weight = tf.get_variable('inception_b_4_4_weight', [7, 1, 224, 256],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        inception_b_conv4 = tf.nn.conv2d(inception_b_conv4, inception_b_4_4_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_b_conv4 = lrelu(inception_b_conv4)
        inception_b_conv4 = tf.contrib.layers.batch_norm(inception_b_conv4, decay=0.9997, epsilon=0.001)

        Inception_B = tf.concat([inception_b_conv1,inception_b_conv2,
                                 inception_b_conv3,inception_b_conv4],3,name=scope.name)
        print(Inception_B)


    # Reduction-B start!
    with tf.variable_scope('Reduction-B') as scope:

        reduction_b_1 = tf.nn.max_pool(Inception_B,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

        reduction_b_2_weight = tf.get_variable('reduction_b_2_weight', [1, 1, 1024, 192],
                                               initializer=tf.contrib.layers.xavier_initializer())

        reduction_b_2 = tf.nn.conv2d(Inception_B, reduction_b_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_b_2 = lrelu(reduction_b_2)
        reduction_b_2 = tf.contrib.layers.batch_norm(reduction_b_2, decay=0.9997, epsilon=0.001)


        reduction_b_2_1_weight = tf.get_variable('reduction_b_2_1_weight', [3, 3, 192, 192],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        reduction_b_2 = tf.nn.conv2d(reduction_b_2, reduction_b_2_1_weight, strides=[1, 2, 2, 1], padding='VALID')
        reduction_b_2 = lrelu(reduction_b_2)
        reduction_b_2 = tf.contrib.layers.batch_norm(reduction_b_2, decay=0.9997, epsilon=0.001)
        print(reduction_b_2)

        reduction_b_3_1_weight = tf.get_variable('reduction_b_3_1_weight', [1, 1, 1024, 256],initializer=tf.contrib.layers.xavier_initializer())
        reduction_b_3 = tf.nn.conv2d(Inception_B, reduction_b_3_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_b_3 = lrelu(reduction_b_3)
        reduction_b_3 = tf.contrib.layers.batch_norm(reduction_b_3, decay=0.9997, epsilon=0.001)

        reduction_b_3_2_weight = tf.get_variable('reduction_b_3_2_weight', [1, 7, 256, 256],initializer=tf.contrib.layers.xavier_initializer())
        reduction_b_3 = tf.nn.conv2d(reduction_b_3, reduction_b_3_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_b_3 = lrelu(reduction_b_3)
        reduction_b_3 = tf.contrib.layers.batch_norm(reduction_b_3, decay=0.9997, epsilon=0.001)

        reduction_b_3_3_weight = tf.get_variable('reduction_b_3_3_weight', [7, 1, 256, 320],initializer=tf.contrib.layers.xavier_initializer())
        reduction_b_3 = tf.nn.conv2d(reduction_b_3, reduction_b_3_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        reduction_b_3 = lrelu(reduction_b_3)
        reduction_b_3 = tf.contrib.layers.batch_norm(reduction_b_3, decay=0.9997, epsilon=0.001)

        reduction_b_3_4_weight = tf.get_variable('reduction_b_3_4_weight', [3, 3, 320, 320],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        reduction_b_3 = tf.nn.conv2d(reduction_b_3, reduction_b_3_4_weight, strides=[1, 2, 2, 1], padding='VALID')
        reduction_b_3 = lrelu(reduction_b_3)
        reduction_b_3 = tf.contrib.layers.batch_norm(reduction_b_3, decay=0.9997, epsilon=0.001)

        Reduction_B = tf.concat([reduction_b_1,reduction_b_2,reduction_b_3],3,name=scope.name)
        print(Reduction_B)

    # Inception-C start!
    with tf.variable_scope('Inception-C') as scope:

        inception_c_1 = tf.nn.avg_pool(Reduction_B, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        inception_c_1_weight = tf.get_variable('inception_c_1_weight', [1, 1, 1536, 256],
                                               initializer=tf.contrib.layers.xavier_initializer())

        inception_c_1 = tf.nn.conv2d(inception_c_1, inception_c_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_1 = lrelu(inception_c_1)
        inception_c_1 = tf.contrib.layers.batch_norm(inception_c_1, decay=0.9997, epsilon=0.001)

        inception_c_2_weight = tf.get_variable('inception_c_2_weight', [1, 1, 1536, 256],
                                               initializer=tf.contrib.layers.xavier_initializer())

        inception_c_2 = tf.nn.conv2d(Reduction_B, inception_c_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_2 = lrelu(inception_c_2)
        inception_c_2 = tf.contrib.layers.batch_norm(inception_c_2, decay=0.9997, epsilon=0.001)

        inception_c_3_weight = tf.get_variable('inception_c_3_weight', [1, 1, 1536, 384],
                                               initializer=tf.contrib.layers.xavier_initializer())

        inception_c_3 = tf.nn.conv2d(Reduction_B, inception_c_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_3 = lrelu(inception_c_3)
        inception_c_3 = tf.contrib.layers.batch_norm(inception_c_3, decay=0.9997, epsilon=0.001)


        inception_c_3_1_weight = tf.get_variable('inception_c_3_1_weight', [1, 3, 384, 256],
                                               initializer=tf.contrib.layers.xavier_initializer())

        inception_c_3_1 = tf.nn.conv2d(inception_c_3, inception_c_3_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_3_1 = lrelu(inception_c_3_1)
        inception_c_3_1 = tf.contrib.layers.batch_norm(inception_c_3_1, decay=0.9997, epsilon=0.001)

        inception_c_3_2_weight = tf.get_variable('inception_c_3_2_weight', [3, 1, 384, 256],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        inception_c_3_2 = tf.nn.conv2d(inception_c_3, inception_c_3_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_3_2 = lrelu(inception_c_3_2)
        inception_c_3_2 = tf.contrib.layers.batch_norm(inception_c_3_2, decay=0.9997, epsilon=0.001)

        inception_c_3 = tf.concat([inception_c_3_1,inception_c_3_2],3)



        inception_c_4_weight = tf.get_variable('inception_c_4_weight', [1, 1, 1536, 384],
                                               initializer=tf.contrib.layers.xavier_initializer())

        inception_c_4 = tf.nn.conv2d(Reduction_B, inception_c_4_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_4 = lrelu(inception_c_4)
        inception_c_4 = tf.contrib.layers.batch_norm(inception_c_4, decay=0.9997, epsilon=0.001)

        inception_c_4_1_weight = tf.get_variable('inception_c_4_1_weight', [1, 3, 384, 448],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        inception_c_4 = tf.nn.conv2d(inception_c_4, inception_c_4_1_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_4 = lrelu(inception_c_4)
        inception_c_4 = tf.contrib.layers.batch_norm(inception_c_4, decay=0.9997, epsilon=0.001)

        inception_c_4_2_weight = tf.get_variable('inception_c_4_2_weight', [3, 1, 448, 512],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        inception_c_4 = tf.nn.conv2d(inception_c_4, inception_c_4_2_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_4 = lrelu(inception_c_4)
        inception_c_4 = tf.contrib.layers.batch_norm(inception_c_4, decay=0.9997, epsilon=0.001)

        inception_c_4_3_weight = tf.get_variable('inception_c_4_3_weight', [3, 1, 512, 256],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        inception_c_4_1 = tf.nn.conv2d(inception_c_4, inception_c_4_3_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_4_1 = lrelu(inception_c_4_1)
        inception_c_4_1 = tf.contrib.layers.batch_norm(inception_c_4_1, decay=0.9997, epsilon=0.001)

        inception_c_4_4_weight = tf.get_variable('inception_c_4_4_weight', [1, 3, 512, 256],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        inception_c_4_2 = tf.nn.conv2d(inception_c_4, inception_c_4_4_weight, strides=[1, 1, 1, 1], padding='SAME')
        inception_c_4_2 = lrelu(inception_c_4_2)
        inception_c_4_2 = tf.contrib.layers.batch_norm(inception_c_4_2, decay=0.9997, epsilon=0.001)

        inception_c_4 = tf.concat([inception_c_4_1,inception_c_4_2],3)

        Inception_C = tf.concat([inception_c_1,inception_c_2,inception_c_3,inception_c_4],3,name=scope.name)

        print(Inception_C)

    #Average Pooling
    with tf.variable_scope('Average-Pooling') as scope:

        average_pooling = tf.nn.avg_pool(Inception_C,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID',name=scope.name)
    print(average_pooling)

    reshape = tf.reshape(average_pooling,[-1,1536])

    with tf.variable_scope('fc') as scope:

        fc_weight = tf.get_variable("fc_weight", shape=[1536, 120],
                         initializer=tf.contrib.layers.xavier_initializer())
        fc_bias = tf.Variable(tf.random_normal([120]))
        fc = lrelu(tf.matmul(reshape, fc_weight) + fc_bias)
        fc = tf.nn.dropout(fc, keep_prob=keep_prob,name=scope.name)

    hypothesis = tf.nn.softmax(fc)
    print(hypothesis)

    return hypothesis
