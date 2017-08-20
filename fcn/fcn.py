import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import dataload as data

# load train , test dataset
train_image, test_image = data.read_image()
train_annotation, test_annotation = data.read_annotation()


# set parameter
batch_size = 10
training_epoach = 500
learning_rate = 0.001
number_train = train_image.shape[0]

# set placeholder
X = tf.placeholder(tf.float32,shape=[None,224,224,3])
Y = tf.placeholder(tf.int32,shape=[None,224,224,1])
keep_prob = tf.placeholder(tf.float32)


def upsampling_convolution(x, W, b, output_shape=None, stride = 2):

    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    print('output_shape:',output_shape)
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def vggnet():
    # conv_1
    weight_1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
    conv_1 = tf.nn.conv2d(X, weight_1, strides=[1, 1, 1, 1], padding='SAME')
    conv_1 = tf.nn.relu(conv_1)
    conv_1 = tf.nn.dropout(conv_1,keep_prob=keep_prob)
    print(conv_1)

    weight_11 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    conv_11 = tf.nn.conv2d(conv_1, weight_11, strides=[1, 1, 1, 1], padding='SAME')
    conv_11 = tf.nn.relu(conv_11)
    print(conv_11)

    conv_11 = tf.nn.max_pool(conv_11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_11 = tf.nn.dropout(conv_11,keep_prob=keep_prob)
    print('pool:', conv_11)

    # conv_2
    weight_2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    conv_2 = tf.nn.conv2d(conv_11, weight_2, strides=[1, 1, 1, 1], padding='SAME')
    conv_2 = tf.nn.relu(conv_2)
    weight_22 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
    conv_22 = tf.nn.conv2d(conv_2, weight_22, strides=[1, 1, 1, 1], padding='SAME')
    conv_22 = tf.nn.relu(conv_22)
    conv_22 = tf.nn.max_pool(conv_22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_22 = tf.nn.dropout(conv_22,keep_prob=keep_prob)
    print(conv_22)

    # conv3
    weight_3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
    conv_3 = tf.nn.conv2d(conv_22, weight_3, strides=[1, 1, 1, 1], padding='SAME')
    conv_3 = tf.nn.relu(conv_3)
    weight_33 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    conv_33 = tf.nn.conv2d(conv_3, weight_33, strides=[1, 1, 1, 1], padding='SAME')
    conv_33 = tf.nn.relu(conv_33)
    weight_333 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    conv_333 = tf.nn.conv2d(conv_33, weight_333, strides=[1, 1, 1, 1], padding='SAME')
    conv_333 = tf.nn.relu(conv_333)
    weight_3333 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    conv_3333 = tf.nn.conv2d(conv_333, weight_3333, strides=[1, 1, 1, 1], padding='SAME')
    conv_3333 = tf.nn.relu(conv_3333)
    conv_3333 = tf.nn.max_pool(conv_3333, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_3333 = tf.nn.dropout(conv_3333,keep_prob=keep_prob)
    print(conv_3333)

    # conv4
    weight_4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
    conv_4 = tf.nn.conv2d(conv_3333, weight_4, strides=[1, 1, 1, 1], padding='SAME')
    conv_4 = tf.nn.relu(conv_4)
    weight_44 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_44 = tf.nn.conv2d(conv_4, weight_44, strides=[1, 1, 1, 1], padding='SAME')
    conv_44 = tf.nn.relu(conv_44)
    weight_444 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_444 = tf.nn.conv2d(conv_44, weight_444, strides=[1, 1, 1, 1], padding='SAME')
    conv_444 = tf.nn.relu(conv_444)
    weight_4444 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_4444 = tf.nn.conv2d(conv_444, weight_4444, strides=[1, 1, 1, 1], padding='SAME')
    conv_4444 = tf.nn.relu(conv_4444)
    conv_4444 = tf.nn.max_pool(conv_4444, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_4444 = tf.nn.dropout(conv_4444,keep_prob=keep_prob)
    print(conv_4444)

    # conv5
    weight_5 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_5 = tf.nn.conv2d(conv_4444, weight_5, strides=[1, 1, 1, 1], padding='SAME')
    conv_5 = tf.nn.relu(conv_5)
    weight_55 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_55 = tf.nn.conv2d(conv_5, weight_55, strides=[1, 1, 1, 1], padding='SAME')
    conv_55 = tf.nn.relu(conv_55)
    weight_555 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    conv_555 = tf.nn.conv2d(conv_55, weight_555, strides=[1, 1, 1, 1], padding='SAME')
    conv_555 = tf.nn.relu(conv_555)
    print('VGG net last shape : ', conv_555)

    return conv_555, conv_4444, conv_3333

def fully_convolutional_network():

    conv_555, conv_4444, conv_3333 = vggnet()

    # FCN
    conv_555 = tf.nn.max_pool(conv_555, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_555 = tf.nn.dropout(conv_555,keep_prob=keep_prob)
    print('FCN start shape : ', conv_555)

    w6 = tf.Variable(tf.random_normal([7, 7, 512, 4096]))
    conv6 = tf.nn.conv2d(conv_555, w6, strides=[1, 1, 1, 1], padding='SAME')
    conv6 = tf.nn.relu(conv6)
    conv6 = tf.nn.dropout(conv6,keep_prob=keep_prob)
    print(conv6)

    w7 = tf.Variable(tf.random_normal([1, 1, 4096, 4096]))
    conv7 = tf.nn.conv2d(conv6, w7, strides=[1, 1, 1, 1], padding='SAME')
    conv7 = tf.nn.relu(conv7)
    conv7 = tf.nn.dropout(conv7,keep_prob=keep_prob)
    print(conv7)

    w8 = tf.Variable(tf.random_normal([1, 1, 4096, 190]))
    conv8 = tf.nn.conv2d(conv7, w8, strides=[1, 1, 1, 1], padding='SAME')
    conv8 = tf.nn.dropout(conv8,keep_prob=keep_prob)
    print(conv8)

    shape = tf.shape(X)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 190])

    w_t1 = tf.Variable(tf.random_normal([4, 4, 512, 190]))
    b_t1 = tf.Variable(tf.random_normal([512]))
    conv_t1 = upsampling_convolution(conv8, w_t1, b_t1, output_shape=tf.shape(conv_4444))
    fuse_1 = tf.add(conv_t1, conv_4444, name='fuse_1')
    print(conv_t1)

    w_t2 = tf.Variable(tf.random_normal([4, 4, 256, 512]))
    b_t2 = tf.Variable(tf.random_normal([256]))
    conv_t2 = upsampling_convolution(fuse_1, w_t2, b_t2, output_shape=tf.shape(conv_3333))
    fuse_2 = tf.add(conv_t2, conv_3333, name='fuse_2')
    print(conv_t2)

    w_t3 = tf.Variable(tf.random_normal([16, 16, 190, 256]))
    b_t3 = tf.Variable(tf.random_normal([190]))
    conv_t3 = upsampling_convolution(fuse_2, w_t3, b_t3, output_shape=deconv_shape3, stride=8)
    print(conv_t3)

    annotation_pred = tf.argmax(conv_t3, dimension=3, name='prediction')
    annotation_pred = tf.expand_dims(annotation_pred, dim=3)
    print(annotation_pred)
    return annotation_pred, conv_t3

annotation_pred , logits = fully_convolutional_network()
loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=tf.squeeze(Y,squeeze_dims=[3]),
                                                                      name='entropy')))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)
sess= tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epoach):
    avg_cost = 0
    total_batch = int(number_train / batch_size)

    for i in range(total_batch):
        train_image = np.reshape(train_image, [-1, 224, 224, 3])
        train_annotation = np.reshape(train_annotation, [-1, 224, 224, 1])


        randidx = np.random.randint(number_train, size=batch_size)

        batch_xs = train_image[randidx, :]
        batch_ys = train_annotation[randidx, :]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:1}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%03d' % (epoch + 1), 'cost =', '{:.5f}'.format(avg_cost))

print('Learning Finished')



pred_image = train_image[0]
pred = sess.run(annotation_pred,feed_dict={X:pred_image,keep_prob:1})
pred = np.squeeze(pred, axis=3)


#prediction part
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.imshow(np.reshape(train_image[0],[224,224,3]))
ax.autoscale(False)
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(pred[0])
ax2.autoscale(False)
plt.show()
