# -*- coding: utf-8 -*-
"""
@author: gwenda

This contains a structure for creating event-based spiking neural networks

HOW TO USE:
    
    to create a network: net = Network()
    
    
Object Contents:

event dictionary: {'type','neuron','time','amplitude'} 
    type: string- force_spike, voltage, learn, last_learn
    neuron: Neuron- where is the event occuring
    time: float- when is the event occuring
    amplitude: float- this is the output amplitude from the synapse
    
synapse dictionary: {'learn_type','in_neuron','out_neuron','weight','lr','length','last_spike_times'}
    learn_type: string- None,'STDP','inhSTDP'
    in_neuron: Neuron- the neuron corresponding to the presynapse
    out_neuron: Neuron- the neuron corresponding to the postsynapse
    weight: float- multiplies by input amplitude to create output amplitude
    lr: float- the learning rate
    length: float- time delay for the voltage to travel the axon
    last_spike_times- a list of the times of the previous events of the synapse
    
neuron_type dictionary: {spiking,leak,sigmoid,learn_delay}
    spiking: float- either spiking (amplitude) or voltage type (None)
    leak: float- sets the decay constant
    sigmoid: tuple (threshold,slope)- this determines spiking probability
    learn_delay: float- how long to wait after spiking to learn
"""
#import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pdb
#import cv2
#import pdb
    

# this just calculates the sigmoid of an array
def sigmoid(X,thresh,slope):
    return 1/(1+np.exp(-1*slope*X+thresh))



# object that manages the event queues
class Controller():
    """
    These queues hold events that need to be processed.
    Events look like {'type','neuron','time','amplitude'}
    Currently, types are: force_spike, voltage, learn, last_learn
    """
    def __init__(self):
        self.queue = []
        self.output = None
    
    def add_event(self,event):

        self.queue.append(event)
        self.queue.sort(key=lambda e:e["time"])
        
        
    def condense_voltage_events(self,event1):
        for event2 in self.queue:
            if event2['type'] == 'voltage':
                if event1['time'] == event2['time'] and event1['neuron'] == event2['neuron']:
                    event2['amplitude'] += event1['amplitude']
                    return True
        return False
    
    # this is the main loop for all neurons
    def run(self, pause_time, predicting, forcing):
        while len(self.queue) > 0:
            if self.output != None:
                out = self.output
                self.output = None
                return out
            
            event = self.queue.pop(0)
            if event['time'] > pause_time:
                # if there are still events after the pause time, erase them
                self.erase_nonlearning_queue()
                return 0 #the id of do nothing

            elif event['type'] == 'force_spike':
                event['neuron'].process_force_spike_input(event)
                
            elif event['type'] == 'voltage':
                if not forcing and event['neuron'].cause_action:
                    event['neuron'].process_voltage_input(event)
                
            elif event['type'] == 'learn':
                if not predicting:
                    event['neuron'].update_weights(event)
                    
            elif event['type'] == 'last_learn':
                self.erase_nonlearning_queue()
                self.output = event['neuron'].id
                if not predicting:
                    event['neuron'].update_weights(event)
                
        return 0 #the id of do nothing
    
                
    # get rid of all the events in queue
    def erase_queue(self):
        self.queue = []
        
    # get rid of events that aren't learning events 
    def erase_nonlearning_queue(self):
        for event in self.queue:
            if event['type'] != 'learn':
                self.queue.remove(event)


        
# object that holds everything to do with a single neuron
class Neuron():
    
    def __init__(self,neuron_type,controller,neuron_id,
                 is_output=False):
        
        self.id = neuron_id
        # neuron type is a dictionary {spiking,leak,sigmoid,lr,learn_delay}
        # spiking float, either spiking (amplitude) or voltage type (None)
        # leak float, sets the decay constant
        # sigmoid tuple (threshold,slope), this determines spiking probability
        # lr float, this is the learning rate
        # learn_delay float, this is the delay before the neuron learns
        # learn should be in the synapse dictionary
        self.type = neuron_type
        self.controller = controller
        self.resting_voltage = 0.0
        # we want to start out at resting voltage
        self.voltage = self.resting_voltage
        # initialize at time = 0
        self.time = 0.0
        
        # set the spike amplitude
        if self.type['spiking'] != None:
            self.spike_amplitude = self.type['spiking']
        
        # set the decay of the voltage
        if self.type['leak'] != None:
            # this is a decay constant with units one over milliseconds
            self.decay = self.type['leak']
            
        # set how long to wait to learn
        self.learn_event_push = neuron_type['learn_delay']
        
        
        # initialize the different synapses
        self.input_synapses = []
        self.output_synapses = [] 
        
        # certain neurons cause actions
        self.cause_action = is_output
        
        
        self.dopamine = 1
		
    
      
    # a force_spike input forces the neuron to fire
    def process_force_spike_input(self,event):
        spike_time = event['time']
        
        # update voltage to the time of the input
        self.update_voltage(spike_time)
        
        # force an action potential
        self.fire(self.spike_amplitude,forced=True)
           
        
        
    # a voltage input gets integrated from a specific synapse
    def process_voltage_input(self,event):
        spike_time = event['time']
        

        # update voltage to the time of the input - types of leak in this one
        self.update_voltage(spike_time, event)

        # for spiking neurons, decide whether to spike
        if self.type['spiking'] != None:
            # use a sigmoid function to determine probability
            if sigmoid(self.voltage,self.type['sigmoid'][0],
                       self.type['sigmoid'][1]) > random.random():
                # fire off an action potential
                self.fire(self.spike_amplitude)
                # send out a learning event to learn from this spike 

                
                
        # if this neuron doesnt spike, then output the current voltage
        else :
            self.fire(self.voltage)
            

        
        

    # we want to calculate what the voltage will be at this new time
    def update_voltage(self,time,event=None):
        # if there is a leak then you want to decay the voltage
        if self.type['leak'] != None:
            # figure out the time difference
            delta_time = time - self.time
            # decay the voltage toward the reversal potential
            try:
                self.voltage = (self.voltage-self.resting_voltage)*math.exp(-delta_time*self.decay)+self.resting_voltage
            except:
                pdb.set_trace()
            
        # if there is no leak then just set voltage back to rest
        else:
            self.voltage = self.resting_voltage
            
        # update to current time
        self.time = time
        
        # if there is input, add the input*weight to the voltage
        if event != None:
            self.voltage = self.voltage + event['amplitude']

        
        
            
    # this causes the neuron to output some amplitude of voltage
    def fire(self,amplitude,forced=False):
        
        # after a spike the voltage goes back to the resting potential
        self.voltage = self.resting_voltage
        # we have to send the spike to each of the synapses
        for synapse in self.output_synapses:
            # add the last spike to history of the synapse for learning
            synapse['last_spike_times'].append(self.time)
            # create an event in the form of a dictionary
            spike = {'neuron':synapse['out_neuron'], 'time':self.time+synapse['length'],
                     'type':'voltage','amplitude':amplitude*synapse['weight']}
            
            # give the spike to the controller to process
            self.controller.add_event(spike)
            

        # send out a learning event to learn from this spike 
        if self.cause_action == True:
            # if this neuron causes an action, we want to: 
            #     erase whatever events are still in the queue
            #     send off a last learn event that will output this neuron's id
            self.controller.erase_nonlearning_queue()
            learning_event = {'neuron':self,'time':self.time+self.learn_event_push,'type':'last_learn','amplitide':None}
            
        else:
            learning_event = {'neuron':self,'time':self.time+self.learn_event_push,'type':'learn','amplitide':None}
        
        
        self.controller.add_event(learning_event)
        
        
        
        
    # this updates the weights, see learn for types of learning
    def update_weights(self,event):
        # this only gets called when the neuron spikes
        spike_time = event['time']-self.learn_event_push
        for synapse in self.input_synapses:
            if synapse['learn_type'] != None:
                last_spike_times = synapse['last_spike_times']
                

                for i in range(len(last_spike_times)):
                    dt = spike_time - last_spike_times[i]
                    self.learn(synapse,dt)

                    
                    
        #self.normalize_input_weights()
        
        

    # this causes the synapse to change its weight
    # should have multiple learn types
    def learn(self,synapse,dt):
        if synapse['learn_type'] == 'Hebbian':
            if dt > 0 and dt < 1:
                synapse['weight'] += synapse['lr'] * self.dopamine
                    

                
        elif synapse['learn_type'] == 'STDP':
            if dt > -1 and dt < 0:
                dt *=-1
                synapse['weight'] -= synapse['lr'] * (1-(dt/1)) * self.dopamine
                
                    
            elif dt > 0 and dt < 1:
                synapse['weight'] += synapse['lr'] * (1-(dt/1)) * self.dopamine
                 

        # generally don't want the weight to drop below zero     
        #if synapse['weight'] < 0:
        #    synapse['weight'] = 0
            

            
    # normalizes all the weights       
    def normalize_input_weights(self):
        if len(self.input_synapses) > 0:
            # compute the max weight
            max_weight = max([synapse['weight'] for synapse in self.input_synapses])
    
            # divide each weight by max
            for synapse in self.input_synapses:
                if synapse['weight'] > 0:
                    synapse['weight'] /= max_weight-.0000001
             
    
    # this adds an input synapse        
    def add_input(self,synapse):
        self.input_synapses.append(synapse)
        
    
    # this adds an output synapse
    def add_output(self,synapse):
        self.output_synapses.append(synapse)
        
    # this just sets the voltage back to resting voltage
    def reset_voltage(self):
        self.voltage = self.resting_voltage



# object that handles creating a network of connected neurons
class Network():
    
    def __init__(self):
        
        self.controller = Controller()
        
        # initialize the array of the grouped neurons
        self.neuron_groups = []
        
        
        
    # add a neuron to a specified group
    def add_neuron(self, neuron_type, neuron_group, neuron_id=random.randrange(10,100),is_output=False):
            
        # create the neuron
        neuron = Neuron(neuron_type,self.controller,neuron_id,
                        is_output=is_output)
        # add it to the group
        neuron_group.append(neuron)
        return neuron
        
    
    # connect one neuron to the other
    def make_synapse(self, learn_type, in_neuron, out_neuron, weight, lr, time_delay):
        # last val is times of spikes that have come through this synapse
        synapse = {'learn_type':learn_type,'in_neuron':in_neuron,
                   'out_neuron':out_neuron,'weight':weight,'lr':lr,
                   'length':time_delay,'last_spike_times':[]}
        in_neuron.add_output(synapse)
        out_neuron.add_input(synapse)
		
        
    # causes neurons of a group to fire
    # adds extra time on if spike value is above 1
    def fire_neurons(self, neurons, spikes, time):
        for i in range(len(neurons)):
            if spikes[i] != 0:
                self.cause_neuron_spikes(neurons[i],time+(spikes[i]-1)*.1)
            
		
        
    # this uses a list of times to cause events
    def cause_neuron_spikes(self,neuron,time):
        spike = {'neuron':neuron, 'time': time+random.random()/100.0, 
                 'type':'force_spike','amplitude':neuron.spike_amplitude}
        self.controller.add_event(spike)
               
    
    # make a group of neurons
    def make_group(self, num_neurons, neuron_type):
        neuron_group = []
        for n in range(num_neurons):
            self.add_neuron(neuron_type,neuron_group)
        self.neuron_groups.append(neuron_group)
        return neuron_group
    
    
    # make a group of neurons
    def make_empty_group(self):
        neuron_group = []
        self.neuron_groups.append(neuron_group)
        return neuron_group
    
    
    # connects group1 to group2, uses random weights
    # if weights are specified, must be in the form of lists of group1 weights
    # for each of group2 neurons
    def connect_groups(self, group1, group2, time_delay, synapse_type="excitatory", 
                       learn_type=None, lr=.01, connectivity=1, weights=None):
        
        for i,neuron1 in enumerate(group1):
            for j,neuron2 in enumerate(group2):
                if weights == None:
                    weight = random.random()
                    if synapse_type == 'inhibitory':
                        weight = -1
                else:
                    weight = weights[j][i]

                self.make_synapse(learn_type,neuron1,neuron2,weight,lr,time_delay)
                
                
    
    #sets the ids of an entire group of neurons based on input
    def set_ids_of_group(self,group,ids):
        for neuron,n_id in zip(group,ids):
            neuron.id = n_id    
            
    # adds (or subtracts) dopamine from a group of neurons
    def flood_with_dopamine(self,neuron_group,reward):
        for neuron in neuron_group:
            neuron.dopamine = reward
            
    # adds (or subtracts) dopamine from all neurons
    def global_dopamine_flood(self,reward):
        for neuron_group in self.neuron_groups:
            self.flood_with_dopamine(neuron_group, reward)
            
    # gets rid of synapses with really small weights
    def purge_synapses(self):
        for group in self.neuron_groups:
            for neuron in group:
                for synapse in neuron.input_synapses:
                    if synapse['weight'] < .0001:
                        neuron.input_synapses.remove(synapse)
    
    # gets rid of neurons that don't have synapses to other neurons (unless output)
    def purge_neurons(self):
        for group in self.neuron_groups:
            for neuron in group:
                if not neuron.cause_action and len(neuron.output_synapses) == 0:
                    group.remove(neuron)
        
    # resets the network for another round of training
    def reset(self):
        self.controller.erase_queue()
        for group in self.neuron_groups:
            for neuron in group:
                neuron.reset_voltage()
                neuron.time = 0
                for synapse in neuron.input_synapses:
                    synapse['last_spike_times'] = []
                    
                        
        
    #run the network
    def run(self,total_time,predicting=False,force_learn=False):
                
        output_neuron_id = self.controller.run(total_time,predicting,force_learn)
        
        return output_neuron_id





