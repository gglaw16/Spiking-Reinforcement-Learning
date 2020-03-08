#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This should use a spiking neural network to play cart pole
"""

import gym
#import numpy as np
#from PIL import Image
import multipurposeNetwork as mpNet
import scipy.stats
import random
import copy



def make_network(spike_amp,threshold,slope,learn_delay,axon_len,lr):  
    '''
    weights = [[[0,0,0,0,0],[0,0,0,0,0]], #position
               [[0,0],[0,0]], #cart velocity
               [[0,0],[0,0]], #pole angle
               [[0,0],[0,0]]] #pole velocity
    '''
    net = mpNet.Network()
    
    neuron_type_spiking = {'spiking':spike_amp,'leak':None,
                           'sigmoid':(threshold,slope),
                           'learn_delay':learn_delay}
    
    neuron_type_nonspiking = {'spiking':None,'leak':None,
                              'sigmoid':(threshold,slope),
                              'learn_delay':learn_delay}
    
    position_place_cells = net.make_group(5,neuron_type_spiking)
    
    cart_velocity = net.make_group(2,neuron_type_nonspiking)
    
    pole_angle = net.make_group(2,neuron_type_nonspiking)
    
    net.set_ids_of_group(pole_angle,[101,102])
    
    pole_velocity = net.make_group(2,neuron_type_nonspiking)
    
    hidden_layer = net.make_group(4,neuron_type_spiking)
    
    output_layer = net.make_empty_group()
    
    # make the three output neurons
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=0,is_output=True)
    
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=1,is_output=True)

    
    # forward connections
    
    net.connect_groups(position_place_cells,hidden_layer,axon_len,
                       learn_type='Hebbian',lr=lr)
    net.connect_groups(cart_velocity,hidden_layer,axon_len,
                       learn_type='Hebbian',lr=lr)
    net.connect_groups(pole_angle,hidden_layer,axon_len,
                       learn_type='Hebbian',lr=lr)
    net.connect_groups(pole_velocity,hidden_layer,axon_len,
                       learn_type='Hebbian',lr=lr)
    
    net.connect_groups(hidden_layer,output_layer,axon_len,
                       learn_type='STDP',lr=lr)
    
    return net,spike_amp,threshold,slope,learn_delay,axon_len,lr



def make_weighted_network(spike_amp,threshold,slope,axon_len,weights):  
    
    net = mpNet.Network()
    
    neuron_type_spiking = {'spiking':spike_amp,'leak':None,
                           'sigmoid':(threshold,slope),
                           'learn_delay':.1}
    
    neuron_type_nonspiking = {'spiking':None,'leak':None,
                              'sigmoid':(threshold,slope),
                              'learn_delay':.1}
    
    position_place_cells = net.make_group(5,neuron_type_spiking)
    
    cart_velocity = net.make_group(2,neuron_type_nonspiking)
    
    pole_angle = net.make_group(2,neuron_type_nonspiking)
    
    net.set_ids_of_group(pole_angle,[101,102])
    
    pole_velocity = net.make_group(2,neuron_type_nonspiking)
    
    hidden_layer = net.make_group(4,neuron_type_spiking)
    
    output_layer = net.make_empty_group()
    
    # make the three output neurons
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=0,is_output=True)
    
    net.add_neuron(neuron_type=neuron_type_spiking,
                   neuron_group=output_layer,neuron_id=1,is_output=True)

    
    # forward connections
    
    net.connect_groups(position_place_cells,hidden_layer,axon_len,
                       learn_type=None,weights=weights[0])
    net.connect_groups(cart_velocity,hidden_layer,axon_len,
                       learn_type=None,weights=weights[1])
    net.connect_groups(pole_angle,hidden_layer,axon_len,
                       learn_type=None,weights=weights[2])
    net.connect_groups(pole_velocity,hidden_layer,axon_len,
                       learn_type=None,weights=weights[3])
    
    net.connect_groups(hidden_layer,output_layer,axon_len,
                       learn_type=None,weights=weights[4])
    
    return net


            
def run_network(net):
    
    
    
    position_place_cells = net.neuron_groups[1] #-.8 to .8
    
    cart_velocity = net.neuron_groups[2] # -2.5 to 2.5
    
    pole_angle = net.neuron_groups[3] #-.25 to .25
    
    pole_velocity = net.neuron_groups[4] #-3 to 3
        
    output_layer = net.neuron_groups[6]
    
    move_left = output_layer[0]
    
    move_right = output_layer[1]
    
    env = gym.make('CartPole-v1')  
    
    for i_episode in range(500):
        #print("Episode %d"%i_episode)
        observation = env.reset()
        done = False
        time = 0
        while not done:
            #env.render()
            pos = observation[0]
            place = [[time+.4-scipy.stats.norm(-.8, .1).pdf(pos)], 
                     [time+.4-scipy.stats.norm(-.4, .1).pdf(pos)],
                     [time+.4-scipy.stats.norm(0, .1).pdf(pos)],
                     [time+.4-scipy.stats.norm(.4, .1).pdf(pos)],
                     [time+.4-scipy.stats.norm(.8, .1).pdf(pos)]]

            
            net.fire_neurons(position_place_cells,place)
            
            if observation[1]<0:
                net.input_image([cart_velocity[0]],[observation[1]*-10],time)
            else:
                net.input_image([cart_velocity[1]],[observation[1]*10],time)
                
            if observation[2]<0:
                net.input_image([pole_angle[0]],[observation[2]*-10],time)
            else:
                net.input_image([pole_angle[1]],[observation[2]*10],time)
                
            if observation[3]<0:
                net.input_image([pole_velocity[0]],[observation[3]*-10],time)
            else:
                net.input_image([pole_velocity[1]],[observation[3]*10],time)

            action = net.run(time+1)
            
            if action == -1:
                action = env.action_space.sample()
                if action == 0:
                    net.cause_neuron_spikes(move_left,[time+.9])
                else:
                    net.cause_neuron_spikes(move_right,[time+.9])
            
            observation, reward, done, info = env.step(action)
            
            edge = -2+abs(observation[0])
            if edge < 0:
                edge = 0
            
            reward = (.1-abs(observation[2]))*10
            
            net.flood_with_dopamine(output_layer, reward)
            print(observation)
            
            time += 1
            
        print(time)
        net.reset_events()

    net.print_weights()


def run_once(net):
    position_place_cells = net.neuron_groups[1]
    
    cart_velocity = net.neuron_groups[2]
    
    pole_angle = net.neuron_groups[3]
    
    pole_velocity = net.neuron_groups[4]
        
    output_layer = net.neuron_groups[6]
    
    move_left = output_layer[0]
    
    move_right = output_layer[1]
    
    env = gym.make('CartPole-v1')  
    
    #print("Episode %d"%i_episode)
    observation = env.reset()
    done = False
    time = 0
    while not done:
        #env.render()
        pos = observation[0]
        place = [[time+.4-scipy.stats.norm(-.8, .1).pdf(pos)], 
                 [time+.4-scipy.stats.norm(-.4, .1).pdf(pos)],
                 [time+.4-scipy.stats.norm(0, .1).pdf(pos)],
                 [time+.4-scipy.stats.norm(.4, .1).pdf(pos)],
                 [time+.4-scipy.stats.norm(.8, .1).pdf(pos)]]

            
        net.fire_neurons(position_place_cells,place)
        
        if observation[1]<0:
            net.input_image([cart_velocity[0]],[observation[1]*-1],time)
        else:
            net.input_image([cart_velocity[1]],[observation[1]],time)
                
        if observation[2]<0:
            net.input_image([pole_angle[0]],[observation[2]*-10],time)
        else:
            net.input_image([pole_angle[1]],[observation[2]*10],time)
            
        if observation[3]<0:
            net.input_image([pole_velocity[0]],[observation[3]*-1],time)
        else:
            net.input_image([pole_velocity[1]],[observation[3]],time)

        action = net.run(time+1)
            
        if action == -1:
            action = env.action_space.sample()
            if action == 0:
                net.cause_neuron_spikes(move_left,[time+.9])
            else:
                net.cause_neuron_spikes(move_right,[time+.9])
        

        
        observation, reward, done, info = env.step(action)
            
        time += 1
            
    net.reset_events()


    return time



mutation_amount = .5
rate = .4

def mutate(spike_amp,threshold,slope,axon_len,weights):
    new_weights = copy.deepcopy(weights)
    for a,input1 in enumerate(weights):
        for b,neuron in enumerate(input1):
            for c,weight in enumerate(neuron):
                if random.random() < rate:
                    new_weights[a][b][c] = weight + mutation_amount * random.choice([-1,1])
                if new_weights[a][b][c] < 0:
                    new_weights[a][b][c] = 0
    
    

    spike_amp1=spike_amp
    threshold1=threshold
    slope1=slope
    axon_len1=axon_len
    
    if random.random() < rate:
        spike_amp1=spike_amp + mutation_amount * random.choice([-1,1])
    if spike_amp1 < 0:
        spike_amp1 = .1


    if random.random() < rate:
        threshold1=threshold + mutation_amount * random.choice([-1,1])
    if threshold1 < 0:
        threshold1 = .1
      
    if random.random() < rate:
        slope1=slope + mutation_amount * random.choice([-1,1])
    if slope1 < 0:
        slope1 = 0
        
    if random.random() < rate:
        axon_len1=axon_len + mutation_amount * random.choice([-1,1])
    if axon_len1 < 0:
        axon_len1 = 0
    
    return spike_amp1,threshold1,slope1,axon_len1,new_weights






def run_generation(net,time,spike_amp,threshold,slope,axon_len,weights):
    best_net = net
    best_time = time
    best_spike_amp = spike_amp
    best_threshold = threshold
    best_slope = slope
    best_axon_len = axon_len
    best_weights = weights
    for i in range(100):
        print('creature test %d'%i)
        spike_amp1,threshold1,slope1,axon_len1,weights1 = \
            mutate(spike_amp,threshold,slope,axon_len,weights)
          
        #print(weights1)
            
        net1 = make_weighted_network(spike_amp1,threshold1,slope1,axon_len1,weights1)
           
        time1 = 0
        for _ in range(20):
            time1 += run_once(net1)
        
        time1 /= 20.0
        
        if time1 > best_time:
            best_net = net1
            best_time = time1
            best_spike_amp = spike_amp1
            best_threshold = threshold1
            best_slope = slope1
            best_axon_len = axon_len1
            best_weights = weights1
            
        print(time1)
        
    
    return best_net, best_time, best_spike_amp, best_threshold, best_slope,\
        best_axon_len, best_weights



def run_weighted_net():
    net,spike_amp,threshold,slope,learn_delay,axon_len,lr =\
        make_network(1,.01,10,.1,.3,.1)
    
    run_network(net)


if __name__ == '__main__':
    
    #run_weighted_net()
    
    
    weights = [[[1, .5, 0, 0, 0], [0, 0, 0, .5, 1],[1, .5, 0, 0, 0], [0, 0, 0, .5, 1]], #position
               [[0,0],[0,0],[0,0],[0,0]], #cart velocity
               [[1,0],[1,0],[0,1],[0,1]], #pole angle
               [[1,0],[1,0],[0,1],[0,1]], #pole velocity
               [[0,0,0,0],[0,0,0,0]]] #hidden layer
    
    
    spike_amp=1
    threshold=1
    slope=10
    axon_len=.1
    
    net = make_weighted_network(spike_amp,threshold,slope,axon_len,weights)
    
    time = 0
    for _ in range(10):
        time += run_once(net)
        
    time /= 10.0
    
    print(time)
    old_time = 0
    for gen in range(1,10):
        print('GEN %d'%gen)
        net,time,spike_amp,threshold,slope,axon_len,weights =\
            run_generation(net,time,spike_amp,threshold,slope,axon_len,weights)
            
        print('')
        print('best time %d'%time)
        print(weights)
        print('')
        
        if time == 500:
            print('done')
            break
        
        if old_time == time:
            rate -= .1
            if rate <= 0:
                rate = 1
        old_time = time
    
        
    
    
    
    