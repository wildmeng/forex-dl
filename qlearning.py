import gym
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys
import random

from collections import defaultdict
from QLEnv import QLEnv
import plotting

matplotlib.style.use('ggplot')

PATH_TRAIN = "./data/train/"
PATH_TEST = "./data/test/"
env = QLEnv(40, path=PATH_TRAIN)

def state_value(Q, state):
    return Q[state[0], state[1], state[2]]


def qLearning(env, num_episodes, discount_factor = 0.95,
                            alpha = 0.7):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    #Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = np.zeros(env.shape+(env.action_space.n,))
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        print("episode %d" % ith_episode)
        state = env.reset()

        for t in itertools.count():
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space.n)
            else:
                act_values = np.argmax(state_value(Q, state))

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(state_value(Q, next_state))
            td_target = reward + discount_factor * state_value(Q, next_state)[best_next_action]
            td_delta = td_target - state_value(Q, state)[action]
            state_value(Q, state)[action] += alpha * td_delta

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats

Q, stats = qLearning(env, 10000)

np.save("Q", Q)

plotting.plot_episode_stats(stats)

#from numpy import genfromtxt
#my_data = genfromtxt('Q.csv', delimiter=',')
