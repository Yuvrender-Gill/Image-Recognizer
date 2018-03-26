import tensorflow as tf
from __future__ import print_function
from ._conv import register_converters as _register_converters
from tensorflow.examples.tutorials.mnsit import input_data

# Initializing the MNIST data set
mnist = input_data.read_data_sets("/temp/data/", one_hot = True)

# Set the learning rate, batch size and display set
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100


# Set the number of hidden layers
n_hidden_1 = 256
n_hidden_2 = 256

# Set the number of input samples and number of classes that we want to classify them to
num_input = 784
num_classes = 10

# Create variables
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Set the weight and biases
weights = {
    'h1': tf.variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.vairable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.airable(tf.random_normal([n_hideen_2, num_classes]))
}

biases = {
    'b1': tf.vairable(tf.random_normal([n_hidden_1])),
    'b2': tf.vairbale(tf.random_normal([n_hidden_2])),
    'out': tf.vairable(tf.random_normal([num_classes]))
}

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1'], biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2'], biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out'] + biases['out'])
    return out_layer


# Constructing the model
logits = neural_net(X)

loss_op = tf.reduce_mean(td.nn.softmax_cross_entropy_with_logits_v2(
    logits = logits, labels = Y
))

# Optimizer

optimizer = td.train.AdamOptimizer(learing_rate = learning_rate)



### Do the acutal training

train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean

#initialize the classifier

init = tf.global_variables_initializer()

if __name__ == "__main__":

    with td.Session() as sess:
        sess.run(init)
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_Y})
            if step %dispay_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict= {X:batch_x, Y:batch_y})
                print("Step " + str(step) + " , Minibatch Loss = " + \
                      "{:0.4f}".format(loss) + ", Training Accuracy = " + \
                      "{:0.3f}".format(acc))
                print(Y)
                print(X)


        print("optimization finished")

        print("Testing Accuracy: ", \
              sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.images})
              )
