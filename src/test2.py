import network, network2
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from sklearn import metrics
import numpy as np

learn = tf.contrib.learn
layers = tf.contrib.layers

import cPickle
import gzip

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [y for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

class MyData(object):
    def __init__(self, training_data):
        self.images = np.array([ i[0] for i in training_data ])
        self.labels = np.array([ i[1] for i in training_data ])

def my_load_data():
    training_data, validation_data, test_data = load_data_wrapper()
    r = (MyData(training_data), MyData(validation_data), MyData(test_data))
    return r

def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution2d(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution2d(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op

def nonconv(training_data, validation_data, test_data):
    feature_columns = learn.infer_real_valued_columns_from_input(
        training_data.images)
    classifier = learn.DNNClassifier([100], feature_columns, model_dir=None, n_classes=10, optimizer=tf.train.FtrlOptimizer(0.3, l2_regularization_strength=0.1), activation_fn=nn.sigmoid, dropout=0.2)

    estimator.SKCompat(classifier).fit(training_data.images,
                   training_data.labels.astype(np.int32),
                   batch_size=10,
                   steps=200000)
    mytuple = (test_data.labels,
                                   list(classifier.predict(test_data.images)))
    score = metrics.accuracy_score(*mytuple)
    print('Accuracy: {0:f}'.format(score))

def conv(training_data, validation_data, test_data):
  classifier = learn.Estimator(model_fn=conv_model)
  classifier.fit(training_data.images,
                 training_data.labels,
                 batch_size=100,
                 steps=20000)
  score = metrics.accuracy_score(test_data.labels,
                                 list(classifier.predict(test_data.images)))
  print('Accuracy: {0:f}'.format(score))

def main(unused_args):
    mytuple = my_load_data()
    conv(*mytuple)

if __name__ == '__main__':
    tf.app.run()
