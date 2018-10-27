import os
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# the current location
current_directory   = os.path.dirname(os.path.realpath(__file__))

# define the path to TensorBoard log files
logPath = current_directory + "/tb_logs/"

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
# download the MNIST data in <current directory>/MNIST_data/
#one_hot=True: one-hot-encoding, means only return the highet probability
mnist = input_data.read_data_sets(current_directory + "/MNIST_data/", one_hot=True)

# make it as default session so we do not need to pass sess
sess = tf.InteractiveSession()

with tf.name_scope("MNIST_Input"):
    # X is placeholder for 28 x 28 image data
    X = tf.placeholder(tf.float32, shape=[None, 784])
    # y_ is a 10 element ventor, it is the predicted probability of each digit class, e.g. [0, 0, 0.12, 0, 0, 0, 0.98, 0, 0.1, 0]
    y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("Input_Reshape"):
    # change the MNIST input data from a list to a 28 x 28 x 1 grayscale value cube, for CNN using
    x_image = tf.reshape(X, [-1, 28, 28, 1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

def weight_variable(shape, name=None):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.087)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.0975, shape=shape)
    return tf.Variable(initial, name=name)

# convolution
def conv2D(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

# max pooling to control overfitting
def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# 1st convolution layer
with tf.name_scope("Conv1"):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], name="weight") # one filter is 5 x 5, 32 features for each filter
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32], name="bias") # Conv1/bias
        variable_summaries(b_conv1)
    
    conv1_wx_b = conv2D(x_image, W_conv1, name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)

    h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1, name="conv2D") + b_conv1, name="relu") # first convolution, use RELU activation
    tf.summary.histogram('h_conv1', h_conv1)
    
    h_pool1 = max_pool_2x2(h_conv1, name="max_pooling") # first max pooling, output is 14 x 14 image ( [28, 28] / 2 = [14, 14])
    tf.summary.histogram('h_pool1', h_pool1)

# 2nd convolution layer
with tf.name_scope("Conv2"):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], name="weight") # one filter is 5 x 5, 64 features for each filter
        variable_summaries(W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64], name="bias")
        variable_summaries(b_conv2)
    conv2_wx_b = conv2D(h_conv1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)

    h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2, name="conv2D") + b_conv2, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    
    h_pool2 = max_pool_2x2(h_conv2, name="max_pooling") # second max pooling, output is 7 x 7 image ( [14, 14] / 2 = [7, 7])
    tf.summary.histogram('h_pool2', h_pool2)

# fully connected layer
with tf.name_scope("FC"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# the final layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# define the model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("cross_entropy"):
    # loss measurement
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("loss_optimization"):
    # loss optimization
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # tf.summary.scalar("train_step", train_step) ## can summarize the loss optimization because "Can't convert Operation 'loss_optimization/Adam' to Tensor"

with tf.name_scope("accuracy"):
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summarize_all = tf.summary.merge_all()
# initialize all variables
sess.run(tf.global_variables_initializer())

# write teh default graph to view its structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# train the model
num_steps = 200
display_every = 10
batch_size = 50
dropout = 0.5

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(batch_size)
    _, summary = sess.run([train_step, summarize_all], feed_dict={X: batch[0], y_:batch[1], keep_prob:dropout})

    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={X:batch[0], y_:batch[1], keep_prob:1.0})
        end_time = time.time()
        print ("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
        tbWriter.add_summary(summary, i)

# test model
print("test accuracy {0:.3}%".format(accuracy.eval(feed_dict={X:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}) * 100.0))