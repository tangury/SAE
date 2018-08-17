import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Reading Mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


#  Parameters
examples_to_show = 10 # finally display 10 pic
mnist_width = 28
n_visible = mnist_width * mnist_width # input layer
n_neuron = [n_visible, 500, 400] # n_visible is input layer size, the numbers after are hidden size neuorn unit nunmbers
corruption_level = 0.3
batch_size = 128
train_epochs = 10
hidden_size = len(n_neuron)-1  # 2
Z = [None]*hidden_size  # Estimated output
cost = [None]*hidden_size
train_op = [None]*hidden_size # training operation


# X as input for each layer
X = tf.placeholder("float", name='X') # demensiononality of input is not defined


# set dictionary for each layer
weights_encoder = dict()
weights_decoder = dict()
biases_encoder = dict()
biases_decoder = dict()
for i in range(hidden_size): # initialize variables for each hidden layer
    W_init_max = 4*np.sqrt(6. / (n_neuron[i]+n_neuron[i+1])) # initialize variables with random values
    # W_init = tf.random_uniform(shape=[n_neuron[i], n_neuron[i+1]], minval=-W_init_max, maxval=W_init_max)
    W_init = tf.random_uniform(shape=[n_neuron[i], n_neuron[i + 1]], minval=-W_init_max, maxval=W_init_max)
    weights_encoder[i] = tf.Variable(W_init)
    weights_decoder[i] = tf.transpose(weights_encoder[i]) # decoder weights are tied with encoder size
    biases_encoder[i] = tf.Variable(tf.random_normal([n_neuron[i+1]]))
    biases_decoder[i] = tf.Variable(tf.random_normal([n_neuron[i]]))




def model(input, W, b, W_prime, b_prime): # One layer model. Output is the estimated output
    Y = tf.nn.sigmoid(tf.matmul(input, W) + b) # hidden state
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime)+ b_prime) # reconstructed input
    return Z


def corruption(input): #corruption of the input
    # mask = np.random.binomial(1, 1-corruption_level, input_data.shape) # mask with several zero at certain position
    mask = np.random.binomial(1, 1 - corruption_level, input.shape)  # mask with several zero at certain position
    corrupted_input = input*mask
    return corrupted_input

def encode(input, W, b, n): # W, b weights encoder and biases encoder, X is the input , n indicate how many layer encode
    #(n=0: input layer n=1: hidden layer)
    if n == 0:
        Y = input # input layer no encode needed
    else:
        for i in range(n):
            Y = tf.nn.sigmoid(tf.add(tf.matmul(input, W[i]), b[i]))
            input = Y # output become input for next layer encode
        Y = Y.eval() # convert tensor.object to ndarray
    return Y

def decode(input, W_prime, b_prime, n):
    if n ==0:     # when it is zero, no decode needed, original output
        Y = input # input layer
    else:
        for i in range(n):
            Y = tf.nn.sigmoid(tf.add(tf.matmul(input, W_prime[n-i-1]), b_prime[n-i-1]))
            input = Y
            Y = Y.eval() # convert tensor.object to ndarray
        return Y


# build the graph
for i in range(hidden_size): # how many layers need to be trained
    Z[i] = model(X, weights_encoder[i], biases_encoder[i], weights_decoder[i], biases_decoder[i])
    # create cost function
    cost[i] = tf.reduce_mean(tf.square(tf.subtract(X, Z[i])))
    train_op[i] = tf.train.GradientDescentOptimizer(0.02).minimize(cost[i])



#Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables!
    tf.global_variables_initializer().run()
    for j in range(hidden_size): # i start form 0
        encode_trX = encode(trX, weights_encoder, biases_encoder, j) # Encode the original input to the certain layer
        encode_teX = encode(teX, weights_encoder, biases_encoder, j) # Also encode the test data to the certain layer
        for i in range(train_epochs):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)): # Give all the batchs
                input = encode_trX[start:end] # take one batch as input to train
                sess.run(train_op[j], feed_dict={X: corruption(input)}) # trainning step, feed the corrupted input
            print("Layer:", j, i, sess.run(cost[j], feed_dict={X: encode_teX}))  #calculate the loss after one epoch. Cost should be calculated with uncorrupted data
        print("One layer Optimizeation Finished!")
    print("All parameters optimized")


# applying encode and decode over test set
    output = tf.constant(decode(encode(teX[:examples_to_show], weights_encoder, biases_encoder, hidden_size),
                         weights_decoder, biases_decoder, hidden_size))
    final_result = sess.run(output)

# Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize = (10, 2) )
    for i in range(examples_to_show):
        a[0][1].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(final_result[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress(timeout=-1)  #add timeout=-1

