from collections import deque
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from PPOAgent import PPOAgent
from PPOEnv import PPOEnv

'''
PPO Agent with Gaussian policy
'''

def run_episode(env, agent, animate=False, evaluation=False): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False
    while not done:
        obs = obs.astype(np.float32).reshape((1, -1))
        observes.append(obs)
        if evaluation:
            action = agent.control(obs).reshape((1, -1)).astype(np.float32)
        else:
            action = agent.get_action(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        infos.append(info)
        
    return (np.concatenate(observes), np.concatenate(actions), np.array(rewards, dtype=np.float32), infos)


def run_policy(env, agent, episodes, evaluation=False): # collect trajectories. if 'evaluation' is ture, then only mean value of policy distribution is used without sampling.
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, infos = run_episode(env, agent, evaluation=evaluation)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos}
        trajectories.append(trajectory)
    return trajectories

    
def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    
    return observes, actions, returns

def compute_returns(trajectories, gamma=0.995): # Add value estimation for each trajectories
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        returns = np.zeros_like(rewards)
        g = 0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + gamma*g
            returns[t] = g
        trajectory['returns'] = returns
        

seed = 0
env = PPOEnv()
np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed=seed)

obs_dim = env.observation_space.shape[0]
n_act = 7 #config: act_dim #env.action_space.n

agent = PPOAgent(obs_dim, n_act, epochs=10,
                          hdim=64, lr=3e-4, max_std=1.0,
                          clip_range=0.3, seed=seed)

avg_return_list = deque(maxlen=10)
avg_loss_list = deque(maxlen=10)

episode_size = 1
batch_size = 64
nupdates = 600

for update in range(nupdates+1):
    trajectories = run_policy(env, agent, episode_size)
    compute_returns(trajectories)
    observes, actions, returns = build_train_set(trajectories)

    pol_loss, kl, entropy = agent.update(observes, actions, returns, batch_size=batch_size)

    avg_loss_list.append(pol_loss)
    avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
    if (update%10) == 0:
        print('[{}/{}] return : {:.3f},  policy loss : {:.3f}, policy kl : {:.5f}, policy entropy : {:.3f}'.format(
            update, nupdates, np.mean(avg_return_list), np.mean(avg_loss_list), kl, entropy))
        
    if (np.mean(avg_return_list) > 90): # Threshold return to success cartpole
        print('[{}/{}] return : {:.3f}, policy loss : {:.3f}'.format(update,nupdates, np.mean(avg_return_list), np.mean(avg_loss_list)))
        print('The problem is solved with {} episodes'.format(update*episode_size))
        break