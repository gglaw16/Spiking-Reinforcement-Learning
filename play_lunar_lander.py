#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This should use a spiking neural network to play lunar lander
"""



"""
LunarLander-v2
Observation , size 8:
min  -1.02 -0.5 -2.3  -2.1 -4.2 -7.3  0  0
max   1.02  1.7  2.3   0.6  3.8  7.3  1  1

"""

import gym
import random
import multipurposeNetwork_inworks as mpNet
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from collections import deque

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)



def create_input_from_state(state):
    input_spikes = []
    normalized_state = []
    normalized_state.append(state[0]/1.02)
    normalized_state.append((state[1]-.825)/.875)
    normalized_state.append(state[2]/2.3)
    normalized_state.append((state[3]+.75)/1.35)
    normalized_state.append((state[4]+.2)/4)
    normalized_state.append(state[5]/7.3)
    normalized_state.append(state[6]*2-1)
    normalized_state.append(state[7]*2-1)

    for i in range(8):
        input_spikes.append(scipy.stats.norm(-1,.5).pdf(normalized_state[i])*4)
        input_spikes.append(scipy.stats.norm(0,.5).pdf(normalized_state[i])*4)
        input_spikes.append(scipy.stats.norm(1,.5).pdf(normalized_state[i])*4)

    
    return [round(i) if i<1 else i for i in input_spikes]



            
def predict(agent,states):
    
    input_layer = agent.neuron_groups[0]

    actions = []
    
    agent.reset()
    time = 0
    
    for e in range(len(states)):
        
        state = states[e]
        state = np.reshape(state, (1, 8))

        spikes = create_input_from_state(state[0])
            
        agent.fire_neurons(input_layer,spikes,time)

        action = agent.run(time+1,predicting=True)
        
        actions.append(action)
        time+=1
        
    return np.array(actions)


def fit(agent,states,targets,rewards):
    
    input_layer = agent.neuron_groups[0]
    output_layer = agent.neuron_groups[-1]


    actions = []
    
    agent.reset()
    time = 0
    
    for e in range(len(states)):
        agent.global_dopamine_flood(rewards[e])
        
        state = states[e]
        state = np.reshape(state, (1, 8))

        spikes = create_input_from_state(state[0])
            
        agent.fire_neurons(input_layer,spikes,time)
        agent.fire_neurons([output_layer[targets[e]]],[1],time+.5)

        action = agent.run(time+1,force_learn=True)
        
        actions.append(action)
        time += 1
        
    return np.array(actions)


class Network:

    """ Implementation of learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        spike_amp=1
        threshold=4
        slope=10
        axon_len=.1
        leak = 5
        learn_delay = .1
            
        net = mpNet.Network()
        
        neuron_type = {'spiking':spike_amp,'leak':leak,
                               'sigmoid':(threshold,slope),
                               'learn_delay':learn_delay}
    
        # we want to make this 24 because you have 3 for each of the 8
        input_layer = net.make_group(24,neuron_type)
        
        hidden_layer = net.make_group(50,neuron_type)
        hidden_layer2 = net.make_group(50,neuron_type)
        
        output_layer = net.make_empty_group()
        
        # make the four output neurons
        
        net.add_neuron(neuron_type=neuron_type,
                       neuron_group=output_layer,neuron_id=0,is_output=True)
        
        net.add_neuron(neuron_type=neuron_type,
                       neuron_group=output_layer,neuron_id=1,is_output=True)
        
        net.add_neuron(neuron_type=neuron_type,
                       neuron_group=output_layer,neuron_id=2,is_output=True)
        
        net.add_neuron(neuron_type=neuron_type,
                       neuron_group=output_layer,neuron_id=3,is_output=True)
        
        
        # forward connections
        
        net.connect_groups(input_layer,hidden_layer,axon_len,
                           learn_type='Hebbian',lr=self.lr)
        
        
        net.connect_groups(hidden_layer,hidden_layer2,axon_len,
                           learn_type='Hebbian',lr=self.lr)

        
        net.connect_groups(hidden_layer2,output_layer,axon_len,
                           learn_type='Hebbian',lr=self.lr)
        
        return net
    


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = predict(self.model,state)
        
        return act_values[0]

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        #actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        #dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = predict(self.model,next_states)
        

        fit(self.model, states, targets, rewards)
        
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            


def train_network(episode):

    loss = []
    agent = Network(env.action_space.n, env.observation_space.shape[0])
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
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        

    return loss


if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    loss = train_network(episodes)
    env.close()
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()









    
        
    
    
    
    