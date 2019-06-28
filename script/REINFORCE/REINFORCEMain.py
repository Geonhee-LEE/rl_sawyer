from collections import deque
import numpy as np
import tensorflow as tf
from REINFORCEAgent import REINFORCEAgent
from REINFORCEEnv import REINFORCEEnv

def run_episode(env, agent): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False
    while not done:
        action = agent.get_action([obs])
        
        next_obs, reward, done, info = env.step(action)
        
        observes.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)

        obs = next_obs
    return np.asarray(observes), np.asarray(actions), np.asarray(rewards), infos

def run_policy(env, agent, episodes): # collect trajectories. if 'evaluation' is ture, then only mean value of policy distribution is used without sampling.
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, infos = run_episode(env, agent)
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
env = REINFORCEEnv()
np.random.seed(seed)
tf.set_random_seed(seed)
env.seed(seed=seed)

obs_dim = env.observation_space.shape[0]
n_act = 7 #config: act_dim #env.action_space.n
agent = REINFORCEAgent(obs_dim, n_act, epochs=5, hdim=32, lr=3e-4,seed=seed)

avg_return_list = deque(maxlen=10)
avg_loss_list = deque(maxlen=10)

episode_size = 1
batch_size = 64
nupdates = 1000

for update in range(nupdates+1):
    #print ('update: ', update)
    trajectories = run_policy(env, agent, episodes=episode_size)
    compute_returns(trajectories)
    observes, actions, returns = build_train_set(trajectories)

    pol_loss = agent.update(observes, actions, returns, batch_size=batch_size)
    
    avg_loss_list.append(pol_loss)
    avg_return_list.append([np.sum(t['rewards']) for t in trajectories])
    if (update%200) == 0:
        print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update, nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
        
    if (np.mean(avg_return_list) > 490): # Threshold return to success cartpole
        print('[{}/{}] policy loss : {:.3f}, return : {:.3f}'.format(update,nupdates, np.mean(avg_loss_list), np.mean(avg_return_list)))
        print('The problem is solved with {} episodes'.format(update*episode_size))
        break