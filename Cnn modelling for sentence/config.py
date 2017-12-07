import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
THIS_DIR = os.path.dirname(os.path.realpath(__file__))


tf.app.flags.DEFINE_string('train_dir', './data/trec_train.csv',
                           'TREC-QA train data path')

tf.app.flags.DEFINE_string('test_dir', './data/trec_test.csv',
                           'TREC-QA test data path')


