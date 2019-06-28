#!/usr/bin/env python

#########################################
# Srikanth Kilaru
# Fall 2018
# MS Robotics, Northwestern University
# Evanston, IL
# srikanthkilaru2018@u.northwestern.edu
##########################################
import sys
import os
import glob
from math import ceil
import argparse
import actionlib
import imutils
from imutils import paths
import copy
import numpy as np
import tensorflow as tf
import logz
import time
import inspect
from multiprocessing import Process
from sklearn import preprocessing
import yaml
import shutil
import ros_env

#====================================================================#
# Utilities
#====================================================================#

def build_mlp(input_placeholder, output_size, scope,
              n_layers, size, activation=tf.tanh,
              output_activation=None):

    with tf.variable_scope(scope):
        #Input layer
        layer = tf.layers.dense(input_placeholder, size,
                                activation=activation,
                                use_bias=True,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        for n in range(n_layers-1):
            layer = tf.layers.dense(layer, size, activation=activation,
                                    use_bias=True,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        # Output fully connected layer 
        output_placeholder = tf.layers.dense(layer, output_size,
                                             activation=output_activation,
                                             use_bias=True,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                
    return output_placeholder

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    
#====================================================================#
# Agent running Policy Gradient algorithm
#====================================================================#

class Agent(object):
    def __init__(self, path, sim_mode=True):
        stream = open(path + "/pg_init.yaml", "r")
        config = yaml.load(stream)
        stream.close()
        
        self.gamma = config['gamma']
        self.n_iter = config['n_iter']
        self.learning_rate = config['learning_rate']
        self.reward_to_go = config['reward_to_go']
        self.normalize_advantages = config['normalize_advantages']
        self.seed = config['seed']
        self.nn_baseline = config['nn_baseline']
        self.n_layers = config['n_layers']
        self.size = config['size']
        self.ob_dim = config['goal_obs_dim'] + config['jnt_obs_dim']
        self.ac_dim = config['act_dim']
        self.max_path_length = config['max_path_length']
        self.min_timesteps_per_batch = config['min_timesteps_per_batch']
        
        print("gamma = ", self.gamma)
        print("n_iter = ", self.n_iter)
        print("learning_rate = ", self.learning_rate)
        print("reward to go = ", self.reward_to_go)
        print("normalize advantages = ", self.normalize_advantages)
        print("seed = ", self.seed)
        print("nn_baseline = ", self.nn_baseline)
        print("n_layers = ", self.n_layers)
        print("size = ", self.size)        
        print("max_path_length = ", self.max_path_length)
        print("min_timesteps_per_batch = ", self.min_timesteps_per_batch)
        print("ob_dim = ", self.ob_dim)
        print("ac_dim = ", self.ac_dim)


def train_PG(logdir, path='.', sim_mode=True):
    
    start = time.time()

    # Initialize the ROS/Sim Environment
    env = ros_env.Env(path, train_mode=True)
    
    # initialize the ROS agent
    agent = Agent(path, sim_mode=sim_mode)

    # Set Up Logger
    setup_logger(logdir, locals())
    
    # Set random seeds
    tf.set_random_seed(agent.seed)
    np.random.seed(agent.seed)

    # Maximum length for episodes
    max_path_length = agent.max_path_length
    
    # Observation and action sizes
    ob_dim = agent.ob_dim
    ac_dim = agent.ac_dim


    """
    Placeholders for batch observations/actions/advantages in policy gradient 
    loss function.
    See Agent.build_computation_graph for notation
    
    sy_ob_no: placeholder for observations
    sy_ac_na: placeholder for actions
    sy_adv_n: placeholder for advantages
    """
    
    sy_ob_no = tf.placeholder(shape=[None, agent.ob_dim],
                              name="ob", dtype=tf.float32)
    
    sy_ac_na = tf.placeholder(shape=[None, agent.ac_dim],
                              name="ac", dtype=tf.float32) 
    
    sy_adv_n = tf.placeholder(shape=[None],
                              name="adv", dtype=tf.float32)
        
    
    """ 
    The policy takes in an observation and produces a distribution 
    over the action space
    Constructs the symbolic operation for the policy network outputs,
    which are the parameters of the policy distribution p(a|s)
    """
    
    # output activations are left linear by not passing any arg
    sy_mean = build_mlp(sy_ob_no, agent.ac_dim,
                        "policy-ddpg",
                        agent.n_layers, agent.size,
                        activation=tf.tanh)
    #print sy_mean.name
    
    sy_logstd = tf.Variable(tf.zeros([1, agent.ac_dim]),
                            dtype=tf.float32, name="logstd")
    
    """ 
    Constructs a symbolic operation for stochastically sampling from 
    the policy distribution
    
    use the reparameterization trick:
    The output from a Gaussian distribution with mean 'mu' and std 
    'sigma' is
    
    mu + sigma * z,         z ~ N(0, I)
    
    This reduces the problem to just sampling z. 
    (use tf.random_normal!)
    """
    
    sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean))
    
    """ 
    We can also compute the logprob of the actions that were
    actually taken by the policy. This is used in the loss function.
    
    Constructs a symbolic operation for computing the log probability 
    of a set of actions that were actually taken according to the policy
    use the log probability under a multivariate gaussian.
    """
        
    action_normalized = (sy_ac_na - sy_mean) / tf.exp(sy_logstd)
    sy_logprob_n = - 0.5 * tf.reduce_sum(tf.square(action_normalized), axis=1)
    
    #=================================================================#
    # Loss Function and Training Operation
    #=================================================================#
    
    loss = -tf.reduce_mean(tf.multiply(sy_logprob_n, sy_adv_n))
    update_op = tf.train.AdamOptimizer(agent.learning_rate).minimize(loss)
    
    #==============================================================#
    # Optional Baseline
    #
    # Define placeholders for targets, a loss function and an update op
    # for fitting a neural network baseline. These will be used to fit the
    # neural network baseline. 
    #===============================================================#
    if agent.nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(sy_ob_no, 1,
                                                   "nn_baseline",
                                                   n_layers=agent.n_layers,
                                                   size=agent.size))
        sy_target_n = tf.placeholder(shape=[None],
                                     name="sy_target_n",
                                     dtype=tf.float32)
        baseline_loss = tf.nn.l2_loss(baseline_prediction - sy_target_n)
        baseline_update_op = tf.train.AdamOptimizer(agent.learning_rate).minimize(baseline_loss)
        
    # tensorflow: config, session, variable initialization
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    

    #====================================================================#
    # Training Loop
    #====================================================================#
    
    total_timesteps = 0
    
    for itr in range(agent.n_iter):
        print("********** Iteration %i ************"%itr)
        itr_mesg = "Iteration started at "
        itr_mesg += time.strftime("%d-%m-%Y_%H-%M-%S")
        print(itr_mesg)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            steps = 0
            while True:
                #time.sleep(0.05)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac,
                              feed_dict={sy_ob_no: ob[None]})
                #print sy_sampled_ac.name (add:0)
                #print sy_ob_no.name (ob:0)
                ac = ac[0]
                acs.append(ac)
                # returns obs, reward and done status
                ob, rew, done = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > agent.max_path_length:
                    break
            path = {"observation" : np.array(obs, dtype=np.float32), 
                    "reward" : np.array(rewards, dtype=np.float32), 
                    "action" : np.array(acs, dtype=np.float32)}            
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > agent.min_timesteps_per_batch:
                break

        total_timesteps += timesteps_this_batch

        '''
        # Build arrays for observation and action for the 
        # policy gradient update by concatenating across paths
        '''
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]


        """
        Monte Carlo estimation of the Q function.        
        Estimates the returns over a set of trajectories.
        
        Store the Q-values for all timesteps and all trajectories in a 
        variable 'q_n', like the 'ob_no' and 'ac_na' above. 
        """
        
        if agent.reward_to_go:
            q_n = []
            for path in paths:
                q = np.zeros(pathlength(path))
                q[-1] = path['reward'][-1]
                for i in reversed(range(pathlength(path) - 1)):
                    q[i] = path['reward'][i] + agent.gamma * q[i+1]
                q_n.extend(q)
        else: 
            q_n = []
            for path in paths: 
                ret_tau = 0
                for i in range(pathlength(path)):
                    ret_tau += (agent.gamma ** i) * path['reward'][i]
                q = np.ones(shape = [pathlength(path)]) * ret_tau
                q_n.extend(q)

        
        """
        Compute advantages by (possibly) subtracting a baseline from the 
        estimated Q values let sum_of_path_lengths be the sum of the 
        lengths of the paths sampled.
        """
        #===========================================================#
        #                       
        # Computing Baselines
        #===========================================================#
        if agent.nn_baseline:
            # If nn_baseline is True, use your neural network to predict
            # reward-to-go at each timestep for each trajectory, and save the
            # result in a variable 'b_n' like 'ob_no', 'ac_na', and 'q_n'.
            #
            # rescale the output from the nn_baseline to match the
            # statistics (mean and std) of the current batch of Q-values.

            b_n = sess.run(baseline_prediction,
                           feed_dict={sy_ob_no: ob_no})
            m1 = np.mean(b_n)
            s1 = np.std(b_n)
            m2 = np.mean(q_n)
            s2 = np.std(q_n)
            b_n = b_n - m1
            b_n = m2 + b_n * (s2/(s1 + 1e-8))
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        
        #=========================================================#
        # Advantage Normalization
        #=========================================================#
        if agent.normalize_advantages:
            # On the next line, implement a trick which is known
            # empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean
            # zero and std=1.
            adv_n = preprocessing.scale(adv_n)
        
        """ 
        Update the parameters of the policy and (possibly) the neural 
        network baseline, which is trained to approximate the value function
        """
        #========================================================#
        # Optimizing Neural Network Baseline
        #========================================================#
        if agent.nn_baseline:
            # If a neural network baseline is used, set up the targets and
            # the inputs for the baseline. 
            # 
            # Fit it to the current batch in order to use for the next
            # iteration. Use the baseline_update_op you defined earlier.
            #
            # Instead of trying to target raw Q-values directly,
            # rescale the targets to have mean zero and std=1.

            target_n = preprocessing.scale(q_n)
            sess.run(baseline_update_op,
                     feed_dict={sy_target_n: target_n,
                                sy_ob_no: ob_no})
            
        #=================================================================#
        # Performing the Policy Update
        #=================================================================#

        # Call the update operation necessary to perform the policy
        # gradient update based on the current batch of rollouts.


        _, after_loss = sess.run([update_op, loss],
                                 feed_dict = {sy_ob_no : ob_no, sy_ac_na : ac_na, sy_adv_n : adv_n})
        
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("After-Loss", after_loss)
        logz.dump_tabular()
        logz.pickle_tf_vars()

        model_file = os.path.join(logdir, "model.ckpt")
        save_path = saver.save(sess, model_file)
        print("Model saved in file: %s" % save_path)
        
    env.close_env_log()
    
def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--sim', action='store_true')
    #parser.add_argument('path', type=str)
    #args = parser.parse_args()
    # path where the python script for agent and env reside
    path = '.' #args.path
    sim = True #args.sim

    dpath = path + '/data'
    if not(os.path.exists(dpath)):
        os.makedirs(dpath)
    logdir = "PG" + '_' + time.strftime("%d-%m-%Y_%H-%M")
    logdir = os.path.join(dpath, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    fname = os.path.basename(sys.argv[0])
    shutil.copyfile('./' + fname, logdir + fname)

    train_PG(logdir, path=path, sim_mode=sim)
    
if __name__ == "__main__":
    main()

