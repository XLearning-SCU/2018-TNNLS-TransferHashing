import tensorflow as tf
import numpy as np

# params = [n_input, n_hidden_1, n_hidden_2, n_bins]

# Create model
def multilayer_perceptron(x, params,scope='mlp'):
    with tf.variable_scope(scope):
        # Network Parameters
        n_input = params[0]
        n_hidden_1 = params[1]
        n_hidden_2 = params[2]
        n_bins = params[3]

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_bins]), name='out')
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
            'b_out': tf.Variable(tf.random_normal([n_bins]), name='b_out')
        }
        reg1 = tf.add_n( [tf.norm(weights[item]) for item in weights ])
        reg2 = tf.add_n( [tf.norm(biases[item]) for item in biases ])
        reg = tf.add(reg1,reg2,name='reg')

        orth1 = tf.square(tf.norm( tf.subtract(tf.matmul(
            a=weights['h1'], b=weights['h1'], transpose_a=True, transpose_b=False),tf.eye(n_hidden_1))))
        orth2 = tf.square(tf.norm( tf.subtract(tf.matmul(
            a=weights['h2'], b=weights['h2'], transpose_a=True, transpose_b=False),tf.eye(n_hidden_2))))
        orth3 = tf.square(tf.norm( tf.subtract(tf.matmul(
            a=weights['out'], b=weights['out'], transpose_a=True, transpose_b=False),tf.eye(n_bins))))
        orth = 0.5 * tf.add_n([orth1,orth2,orth3],name='orth')

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.l2_normalize(layer_1,dim=1)
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.l2_normalize(layer_2,dim=1)
        layer_2 = tf.nn.tanh(layer_2)
        # Output layer with linear activation
        out = tf.matmul(layer_2, weights['out']) + biases['b_out']
        z = tf.nn.tanh(out)
        b = tf.sign(z,name='b')
    return z,b,reg,orth

def multilayer_perceptron_aux(x, params, scope='mlp_aux'):
    with tf.variable_scope(scope):
        # Network Parameters
        n_input = params[0]
        n_hidden_1 = params[1]
        n_hidden_2 = params[2]
        n_bins = params[3]

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='h2'),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_bins]), name='out')
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
            'b_out': tf.Variable(tf.random_normal([n_bins]), name='b_out')
        }
        reg1 = tf.add_n( [tf.norm(weights[item]) for item in weights ])
        reg2 = tf.add_n( [tf.norm(biases[item]) for item in biases ])
        reg = tf.add(reg1,reg2,name='reg')

        # orth1 = tf.square(tf.norm( tf.subtract(tf.matmul(a=weights['h1'], b=weights['h1'], transpose_a=True, transpose_b=False),tf.eye(n_hidden_1))))
        # orth2 = tf.square(tf.norm( tf.subtract(tf.matmul(a=weights['h2'], b=weights['h2'], transpose_a=True, transpose_b=False),tf.eye(n_hidden_2))))
        # orth3 = tf.square(tf.norm( tf.subtract(tf.matmul(a=weights['out'], b=weights['out'], transpose_a=True, transpose_b=False),tf.eye(n_bins))))
        # orth = tf.add_n([orth1,orth2,orth3],name='orth')

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.l2_normalize(layer_1,dim=1)
        layer_1 = tf.nn.tanh(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.l2_normalize(layer_2,dim=1)
        layer_2 = tf.nn.tanh(layer_2)
        # Output layer with linear activation
        out = tf.matmul(layer_2, weights['out']) + biases['b_out']
        out = tf.nn.tanh(out)
    return out,reg
