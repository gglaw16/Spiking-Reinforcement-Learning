#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:48:31 2020

@author: gwenda
"""


#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.


import gym
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import linear, sigmoid
import keras.backend as kb


import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


def hardsigmoid(x):
    return kb.round(sigmoid(x))
    #return sigmoid(4*x)

class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space, weights):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()
        self.model.set_weights(weights)

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=hardsigmoid))
        model.add(Dense(120, activation=hardsigmoid))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
        return model

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

            
    def get_weights(self):
        return self.model.get_weights()


def train_dqn(episode):
    
    scores = []

    weights = np.load('weights_4.npy',allow_pickle=True)
    agent = DQN(env.action_space.n, env.observation_space.shape[0], weights)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            state = next_state
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        scores.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(scores[-100:])

        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        

    return scores


if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 200
    loss = train_dqn(episodes)
    env.close()
    plt.hist(loss[::2],bins=20)
    plt.show()