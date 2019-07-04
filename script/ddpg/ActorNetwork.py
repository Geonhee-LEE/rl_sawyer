import tensorflow as tf
import numpy as np
from collections import deque


class ActorNetwork(object):
    def __init__(self, sess, obs_dim, act_dim, act_min, act_max, lr=1e-4, tau=1e-3, batch_size=64, hdim = 64):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.hdim = hdim
        self.lr = lr
        
        self.tau = tau # Parameter for soft update
        self.batch_size = batch_size
        
        self.seed = 0

        # Actor Network
        self.inputs, self.out = self.create_actor_network(scope='actor')
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network(scope='target_actor')
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor')
        
        # Parameter Updating Operator
        self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]

        # Gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.act_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
#         self.actor_gradients = list(map(lambda x: x/self.batch_size, self.unnormalized_actor_gradients))
        self.actor_gradients = [ unnz_actor_grad/self.batch_size for unnz_actor_grad in self.unnormalized_actor_gradients]

        # Optimizer
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self,scope='network'):
        hid1_size = self.hdim
        hid2_size = self.hdim
        
        with tf.variable_scope(scope): # Two Layer Network
            
            # Input Placeholder
            inputs = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            
            out = tf.layers.dense(inputs, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh, # Tangent Hyperbolic Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            out = tf.layers.dense(out, self.act_dim, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='act')
            out = tf.sigmoid(out)*(self.act_max-self.act_min) + self.act_min
        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
