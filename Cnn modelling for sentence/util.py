import numpy as np
import pandas as pd
import re
from config import FLAGS
from tensorflow.contrib import learn


def calc_acc(self, sess, x, y):
    nbatches = int(len(x) / self.batch_size)

    acc = 0
    for i in range(nbatches):
        acc += sess.run(self.accuracy, feed_dict={
            self.input_x: x[i * self.batch_size:(i + 1) * self.batch_size],
            self.input_y: y[i * self.batch_size:(i + 1) * self.batch_size],
            self.keep_prob: 1.0})
    return acc / nbatches


def clean_str(s):
    # only include alphanumerics
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    # insert spaces in words with apostrophes
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    # insert spaces in special characters
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    # reduce multiple spaces to single spaces
    s = re.sub(r"\s{2,}", " ", s)

    # only include alphanumerics again
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    return s.strip().lower()


def load_data(self):
    train_df = pd.read_csv(FLAGS.train_dir)
    test_df = pd.read_csv(FLAGS.test_dir)

    length = [len(str(x).split(' ')) for x in train_df['data']]
    self.sequence_length = max(length)

    # make one-hot label
    labels = sorted(list(set(train_df['target'].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    y_train_raw = train_df['target'].apply(lambda y: label_dict[y]).tolist()
    y_test_raw = test_df['target'].apply(lambda y: label_dict[y]).tolist()

    # data cleaning
    x_train_raw = train_df['data'].apply(lambda x: clean_str(x)).tolist()
    x_test_raw = test_df['data'].apply(lambda x: clean_str(x)).tolist()

    # build vocab
    vocab_processor = learn.preprocessing.VocabularyProcessor(self.sequence_length)
    self.x_train = np.array(list(vocab_processor.fit_transform(x_train_raw)))
    self.y_train = np.array(y_train_raw)
    self.x_test = np.array(list(vocab_processor.transform(x_test_raw)))
    self.y_test = np.array(y_test_raw)

    # number of total word
    self.vocab_size = len(vocab_processor.vocabulary_)
    self.num_class = self.y_train.shape[1]

    print("sequence length is %s" % self.sequence_length)
    print("vocab size is %s" % self.vocab_size)
    print("number of class is %s" % self.num_class)
    print("number of train data set is %s" % len(self.x_train))
    print("number of test data set is %s" % len(self.x_test))

