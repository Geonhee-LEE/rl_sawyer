from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OrnsteinUhlenbeckActionNoise import OrnsteinUhlenbeckActionNoise
from OrnsteinUhlenbeckActionNoise import NormalActionNoise
from ReplayBuffer import ReplayBuffer
from DDPGEnv import DDPGEnv

tf.reset_default_graph()
sess=tf.Session()

seed = 0
env = DDPGEnv()
np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed=seed)

obs_dim = env.observation_space.shape[0]
print ('obs_dim: ', obs_dim)
action_dim = env.action_space.shape[0]
print ('action_dim: ', action_dim)
action_max = env.action_space.high
action_min = env.action_space.low

# Actor, Critic, Noise(OrnsteinUhlenbeck)
actor = ActorNetwork(sess, obs_dim, action_dim, action_min, action_max)
critic = CriticNetwork(sess, obs_dim, action_dim)
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

sess.run(tf.global_variables_initializer())
actor.update_target_network()
critic.update_target_network()

# Memory Size 10000
replay_buffer = ReplayBuffer(10000, seed)

max_episodes = 5000
max_episode_len = 170
avg_return_list = deque(maxlen=10)

for i in range(max_episodes+1):
    obs = env.reset()

    ep_reward = 0
    ep_ave_max_q = 0

    for j in range(max_episode_len):

        # Added exploration noise
        action = actor.predict(np.reshape(obs, (1, actor.obs_dim))) + actor_noise()

        next_obs, reward, done, info = env.step(action[0])

        replay_buffer.add(np.reshape(obs, (actor.obs_dim,)), np.reshape(action, (actor.act_dim,)), reward,
                          done, np.reshape(next_obs, (actor.obs_dim,)))

        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        if replay_buffer.size() >  actor.batch_size:
            o_batch, a_batch, r_batch, d_batch, no_batch = replay_buffer.sample_batch(actor.batch_size)

            # Calculate targets
            target_q = critic.predict_target(
                no_batch, actor.predict_target(no_batch))

            y_i = []
            for k in range(actor.batch_size):
                if d_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(
                o_batch, a_batch, np.reshape(y_i, (actor.batch_size, 1)))

            ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(o_batch)
            grads = critic.action_gradients(o_batch, a_outs)
            actor.train(o_batch, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

        obs = next_obs
        ep_reward += reward
        if done:            
            break
    avg_return_list.append(ep_reward)
    if (i%1)==0:
        print('[{}/{}] return : {:.3f}, da : {:.3f}'.format(i,max_episodes, np.mean(avg_return_list), np.mean(actor_noise.x_prev)))
    
    if (np.mean(avg_return_list) > 200): # Threshold return to success cartpole
        print('[{}/{}] return : {:.3f}, da : {:.3f}'.format(i,max_episodes, np.mean(avg_return_list), np.mean(actor_noise.x_prev)))
        print('The problem is solved with {} episodes'.format(i))
        break
