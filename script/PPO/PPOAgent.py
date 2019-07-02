from collections import deque
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

class PPOAgent(object):
    def __init__(self, obs_dim, act_dim, clip_range=0.2,
                 epochs=10, lr=3e-5, hdim=64, max_std=1.0,
                 seed=0):
        
        self.seed=0
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.clip_range = clip_range
        
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.max_std = max_std
        
        self._build_graph()
        self._init_session()        

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.variables = tf.global_variables()
            
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.scores_ph = tf.placeholder(tf.float32, (None,), 'scores')

        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        
        # place holder for old parameters
        self.old_std_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_std')
        self.old_mean_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
       

    def _policy_nn(self):
        hid1_size = self.hdim
        hid2_size = self.hdim
        
        # TWO HIDDEN LAYERS
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")
        
        # MEAN FUNCTION
        self.mean = tf.layers.dense(out, self.act_dim,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), 
                                name="mean")
        # UNI-VARIATE
        self.logits_std = tf.get_variable("logits_std",shape=(1,),initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
        self.std = self.max_std*tf.ones_like(self.mean)*tf.sigmoid(self.logits_std) # IMPORTANT TRICK
        
        # SAMPLE OPERATION
        self.sample_action = self.mean + tf.random_normal(tf.shape(self.mean),seed=self.seed)*self.std
        

    def _logprob(self):
        # PROBABILITY WITH TRAINING PARAMETER
        y = self.act_ph 
        mu = self.mean
        sigma = self.std

        self.logp = tf.reduce_sum(-0.5*tf.square((y-mu)/sigma)-tf.log(sigma)- 0.5*np.log(2.*np.pi),axis=1)

        # PROBABILITY WITH OLD (PREVIOUS) PARAMETER
        old_mu_ph = self.old_mean_ph
        old_sigma_ph = self.old_std_ph
                
        self.logp_old = tf.reduce_sum(-0.5*tf.square((y-old_mu_ph)/old_sigma_ph)-tf.log(old_sigma_ph)- 0.5*np.log(2.*np.pi),axis=1)
        

    def _kl_entropy(self):
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean_ph, self.old_std_ph
 
        log_std_old = tf.log(old_std)
        log_std_new = tf.log(std)
        frac_std_old_new = old_std/std

        # KL DIVERGENCE BETWEEN TWO GAUSSIAN
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/std)- 0.5,axis=1)
        self.kl = tf.reduce_mean(kl)
        
        # ENTROPY OF GAUSSIAN
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5*np.log(2*np.pi),axis=1)
        self.entropy = tf.reduce_mean(entropy)


    def _loss_train_op(self):        
        # Proximal Policy Optimization CLIPPED LOSS FUNCTION
        ratio = tf.exp(self.logp - self.logp_old) 
        clipped_ratio = tf.clip_by_value(ratio,clip_value_min=1-self.clip_range,clip_value_max=1+self.clip_range) 
        self.loss = -tf.reduce_mean(tf.minimum(self.scores_ph*ratio,self.scores_ph*clipped_ratio))
        
        # OPTIMIZER 
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)


    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(self.init)


    def get_action(self, obs): # SAMPLE FROM POLICY
        obs = obs.squeeze()
        feed_dict = {self.obs_ph: [obs]}
        sampled_action = self.sess.run(self.sample_action,feed_dict=feed_dict)
        return sampled_action[0]

    
    def control(self, obs): # COMPUTE MEAN
        obs = obs.squeeze()
        feed_dict = {self.obs_ph: [obs]}
        best_action = self.sess.run(self.mean,feed_dict=feed_dict)
        return best_action          
    

    def update(self, observes, actions, scores, batch_size = 128): # TRAIN POLICY
        
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches
        
        old_means_np, old_std_np = self.sess.run([self.mean, self.std],{self.obs_ph: observes}) # COMPUTE OLD PARAMTER
        for e in range(self.epochs):
            observes, actions, scores, old_means_np, old_std_np = shuffle(observes, actions, scores, old_means_np, old_std_np, random_state=self.seed)
            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end,:],
                     self.scores_ph: scores[start:end],
                     self.old_std_ph: old_std_np[start:end,:],
                     self.old_mean_ph: old_means_np[start:end,:],
                     self.lr_ph: self.lr}        
                self.sess.run(self.train_op, feed_dict)
            
        feed_dict = {self.obs_ph: observes,
                 self.act_ph: actions,
                 self.scores_ph: scores,
                 self.old_std_ph: old_std_np,
                 self.old_mean_ph: old_means_np,
                 self.lr_ph: self.lr}             
        loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
        return loss, kl, entropy
    
    
    def close_sess(self):
        self.sess.close()