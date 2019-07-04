import tensorflow as tf
import numpy as np
from collections import deque

class CriticNetwork(object):
    def __init__(self, sess, obs_dim, act_dim, learning_rate=1e-3, tau=1e-3, gamma=0.995, hidden_unit_size = 64):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_unit_size = hidden_unit_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.seed = 0

        # Create the critic network
        self.inputs, self.action, self.out = self.buil_critic_nn(scope='critic')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.buil_critic_nn(scope='target_critic')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic')

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.out))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)
        
    def buil_critic_nn(self,scope='network'):
        hid1_size = self.hidden_unit_size
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope(scope): # Prediction Network / Two layered perceptron / Training Parameters
            inputs = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            action = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
            out = tf.layers.dense(tf.concat([inputs,action],axis=1), hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh, # Tangent Hyperbolic Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='h2')
            out = tf.layers.dense(out, 1, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q')
        return inputs, action, out
    
    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)