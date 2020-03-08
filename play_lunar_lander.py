#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This should use a spiking neural network to play cart pole
"""

import gym
#import numpy as np
#from PIL import Image
import multipurposeNetwork as mpNet
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

def make_network(action_space,state_space):  

    lr = 0.001

    spike_amp=1
    threshold=1
    slope=10
    axon_len=.1
    learn_delay=.1
        
    net = mpNet.Network()
    
    neuron_type_spiking = {'spiking':spike_amp,'leak':None,
                           'sigmoid':(threshold,slope),
                           'learn_delay':learn_delay}
    

    
    input_layer = net.make_group(state_space,neuron_type_spiking)
    
    hidden_layer = net.make_group(15,neuron_type_spiking)
    
    output_layer = net.make_empty_group()
    
    # make the three output neurons
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=0,is_output=True)
    
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=1,is_output=True)
    
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=2,is_output=True)

    
    # forward connections
    
    net.connect_groups(input_layer,hidden_layer,axon_len,
                       learn_type='STDP',lr=lr)

    
    net.connect_groups(hidden_layer,output_layer,axon_len,
                       learn_type='STDP',lr=lr)
    
    return net


            
def run_network(episode):
    

    
    loss = []
    agent = make_network(env.action_space.n, env.observation_space.shape[0])
    input_layer = agent.neuron_groups[1]
    
    for e in range(episode):
   
        
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        time = 0
        for i in range(max_steps):
            agent.input_image(input_layer,state[0],time)
        
            action = agent.run(time+1)

            
            env.render()
            
            next_state, reward, done, _ = env.step(action)
            
            score += reward
                                    
            state = np.reshape(next_state, (1, 8))
                        
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
            
            time += 1
            
        loss.append(score)
       

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        #if is_solved > -200:
        #    print('\n Task Completed! \n')
        #    break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        
    return loss



if __name__ == '__main__':
    
    print(env.observation_space)
    print(env.action_space)
    episodes = 40
    loss = run_network(episodes)
    env.close()
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()

    
        
    
    
    
    