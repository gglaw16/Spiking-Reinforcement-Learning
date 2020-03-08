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
#import cv2
#import pdb
from neuronpy.graphics import spikeplot
    

# this just calculates the sigmoid of an array
def sigmoid(X,thresh,slope):
    return 1/(1+np.exp(-1*slope*X+thresh))


# this is needed if you want to see the output of the neurons over time
class SpikeRecorder():
                       
    def __init__(self):
        # automatically assumes you are recording if you create a recorder
        self.recording = True
        # this is an empty dictionary of the labels of each neuron
        self.labels = {}
        # initially an empty dictionary, but when full, will have structure {neuron:[], neuron2:[]}
        self.recordings = {} 

    # this is used to turn on and off recording        
    def set_recording(self, val):
        self.recording = val

    # this empties the recordings of each neuron        
    def reset(self):
        for neuron in self.recordings:
            self.recordings[neuron] = []

        
    # adds a neuron pair to the dictionary
    def add_neuron(self,neuron, label=None):
        # if there is no label, set the label to the neuron's id number
        if label is None:
            label = "id%d"%neuron.id
        self.labels[neuron] = label
        self.recordings[neuron] = []

        
    # add the events "time" to the appropriate recording
    def process_event(self,neuron,time):
        if not self.recording:
            return
        if neuron in self.recordings:
            self.recordings[neuron].append(time)

            
    # returns the recordings in the format of a list of lists
    def get_spiketrains(self):
        # spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = [] 
        for neuron in self.recordings:
            times = []
            for time in self.recordings[neuron]:
                times.append(time)
            spiketrains.append(times)
        return spiketrains

    
    def plot(self):
        # spiketrains should be all the times of the spikes in a list separated by neuron, a list of lists
        spiketrains = self.get_spiketrains()
                       
        sp = spikeplot.SpikePlot()
        sp.set_markerscale(0.5)

        sp.plot_spikes(spiketrains)
    
    def plot_one_neuron(self,neuron):
        if neuron in self.recordings:
            spiketrain = self.recordings[neuron]
        
        sp = spikeplot.SpikePlot()
        sp.set_markerscale(0.5)

        sp.plot_spikes(spiketrain)


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
        if event['type'] == 'voltage':
            if event['amplitude'] == 0:
                return
            condensed = self.condense_voltage_events(event)
            if condensed == True:
                return
        elif event['type'] == 'last_learn':
            event['type'] = 'learn'
            self.output = event['neuron'].id

        
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
    def run(self, pause_time):
        while len(self.queue) > 0:
            if self.output != None:
                out = self.output
                self.output = None
                return out
            
            event = self.queue.pop(0)
            if event['time'] > pause_time:
                return -1

            elif event['type'] == 'force_spike':
                event['neuron'].process_force_spike_input(event)
                
            elif event['type'] == 'voltage':
                event['neuron'].process_voltage_input(event)
                
            elif event['type'] == 'learn':
                event['neuron'].update_weights(event)
                
            
        return -1
                
    # get rid of all the events in queue
    def erase_queue(self):
        self.queue = []


        
# object that holds everything to do with a single neuron
class Neuron():
    
    def __init__(self,neuron_type,controller,neuron_id,spike_recorder=None,
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
        self.spike_recorder = spike_recorder
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
            self.voltage = (self.voltage-self.resting_voltage)*math.exp(-delta_time*self.decay)+self.resting_voltage
            
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

        # if we are recording, record this spike
        if self.spike_recorder:
            self.spike_recorder.process_event(self,self.time)
        
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
        if self.cause_action == True and not forced:
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
                
                i = 0
                for _ in range(len(last_spike_times)):
                    
                    dt = spike_time - last_spike_times[i]
                    i = self.learn(synapse,dt,i)
                    i += 1
                    
        self.normalize_input_weights()
                    
                    

    # this causes the synapse to change its weight
    # should have multiple learn types
    def learn(self,synapse,dt,i):
        if synapse['learn_type'] == 'Hebbian':
            if dt > -1 and dt < 0:
                dt *=-1
                synapse['weight'] += synapse['lr'] * self.dopamine
                    
            elif dt > 0 and dt < 1:
                synapse['weight'] += synapse['lr'] * self.dopamine
                    
            else :
                synapse['last_spike_times'].pop(i)
                i = i-1
                
                
        elif synapse['learn_type'] == 'STDP':
            if dt > -1 and dt < 0:
                dt *=-1
                synapse['weight'] -= synapse['lr'] * (1-(dt/1)) * self.dopamine
                
                    
            elif dt > 0 and dt < 1:
                synapse['weight'] += synapse['lr'] * (1-(dt/1)) * self.dopamine
                    
            else :
                synapse['last_spike_times'].pop(i)
                i = i- 1
                
        if synapse['weight'] < 0:
            synapse['weight'] = 0
        return i
                


            
    # normalizes all the weights       
    def normalize_input_weights(self):
        if len(self.input_synapses) > 0:
            # compute the max weight
            max_weight = max([synapse['weight'] for synapse in self.input_synapses])
    
            # divide each weight by max
            for synapse in self.input_synapses:
                synapse['weight'] /= max_weight-.0000001
            
            
    
    # this adds an input synapse        
    def add_input(self,synapse):
        self.input_synapses.append(synapse)
        
        
    
    # this adds an output synapse
    def add_output(self,synapse):
        self.output_synapses.append(synapse)



# object that handles creating a network of connected neurons
class Network():
    
    def __init__(self):
        
        self.controller = Controller()

        self.spike_recorder = SpikeRecorder()
        

        # initialize the array of the grouped neurons
        self.neuron_groups = []
        
        # create a main group for non grouped neurons
        self.main_neuron_group = []
        
        # add the main neuron group to the list of groups
        self.neuron_groups.append(self.main_neuron_group)
        
        
    # add a neuron to a specified group
    def add_neuron(self, neuron_type, neuron_group=None,neuron_id=random.randrange(1,100),is_output=False):
        # if we don't input a group, it goes into the main neuron group
        if neuron_group == None:
            neuron_group = self.main_neuron_group
            
        # create the neuron
        neuron = Neuron(neuron_type,self.controller,neuron_id,
                        spike_recorder=self.spike_recorder,is_output=is_output)
        # add it to the group
        neuron_group.append(neuron)
        self.spike_recorder.add_neuron(neuron)
        return neuron
        
    
    # connect one neuron to the other
    def make_synapse(self, learn_type, in_neuron, out_neuron, weight, lr, time_delay):
        # last val is times of spikes that have come through this synapse
        synapse = {'learn_type':learn_type,'in_neuron':in_neuron,
                   'out_neuron':out_neuron,'weight':weight,'lr':lr,
                   'length':time_delay,'last_spike_times':[]}
        in_neuron.add_output(synapse)
        out_neuron.add_input(synapse)
		
        
    # uses lists of firing times to cause neurons of a group to fire
	# outputs has to be the same length as the number of input neurons
    def fire_neurons(self, neurons, time_list):
        for i in range(len(neurons)):
            self.cause_neuron_spikes(neurons[i],time_list[i])
            
            
    # uses lists of voltages separated by a time step to input into neurons
    def input_voltages(self, neurons, voltages_list, time_step):
        for i in range(len(neurons)):
            self.input_voltage_into_neuron(neurons[i],voltages_list[i],time_step)
            
            
    # this just inputs a bunch of voltages at the same time
    def input_image(self,neurons,voltages,time):
        for i in range(len(neurons)):
            spike = {'neuron':neurons[i], 'time': time, 'type':'voltage',
                        'amplitude':voltages[i]}
            self.controller.add_event(spike)
		
        
    # this uses a list of times to cause events
    def cause_neuron_spikes(self,neuron,times):
        for time in times:
            spike = {'neuron':neuron, 'time': time, 'type':'force_spike',
                         'amplitude':neuron.spike_amplitude}
            self.controller.add_event(spike)
               
            
    # this uses a list of voltages to input into a neuron    
    def input_voltage_into_neuron(self,neuron,voltages,time_step):
        time = 0
        for volt in voltages:
            spike = {'neuron':neuron, 'time': time, 'type':'voltage',
                         'amplitude':volt}
            self.controller.add_event(spike)
            time += time_step
           
            
    # makes a group of neurons that are all connected to each other, uses random weights
    def make_connected_group(self, num_neurons, neuron_type, 
                             synapse_type="excitatory", learn_type=None, 
                             connectivity=1, self_connected=True, weights=None):
        conn_group = []
        for n in range(num_neurons):
            self.add_neuron(neuron_type,conn_group)
        
        for i,neuron1 in enumerate(conn_group):
            for j,neuron2 in enumerate(conn_group):
                if self_connected or neuron1 != neuron2:
                    if connectivity > random.random():
                        if weights == None:
                            weight = random.random()
                            if synapse_type == 'inhibitory':
                                weight *= -1
                        else:
                            weight = weights[i][j]
                        self.make_synapse(learn_type,neuron1,neuron2,weight)
        
        self.neuron_groups.append(conn_group)
        return conn_group
    
    
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
                if connectivity > random.random():
                    if weights == None:
                        weight = random.random()
                        if synapse_type == 'inhibitory':
                            weight *= -1
                    else:
                        weight = weights[j][i]

                    self.make_synapse(learn_type,neuron1,neuron2,weight,lr,time_delay)
    

    def set_ids_of_group(self,group,ids):
        for neuron,n_id in zip(group,ids):
            neuron.id = n_id    
            
    def flood_with_dopamine(self,neuron_group,reward):
        for neuron in neuron_group:

            neuron.dopamine = reward
            
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
        
    def reset_events(self):
        self.controller.erase_queue()
        self.spike_recorder.reset()
        
        
    def plot_all(self):
        self.spike_recorder.plot()
        
        
    def plot_neuron_spikes(self,neuron):
        self.spike_recorder.plot_one_neuron(neuron)
        
    def print_weights(self):
        weights = []
        for group in self.neuron_groups:
            each_group = []
            for neuron in group:
                each_neuron = []
                for synapse in neuron.output_synapses:
                    each_neuron.append(synapse['weight'])
                if each_neuron != []:
                    each_group.append(each_neuron)
            if each_group != []:
                weights.append(each_group)
        print(weights)
        
    #run the network
    def run(self,total_time,record=True):
        
        self.spike_recorder.set_recording(record)
        
        output_neuron_id = self.controller.run(pause_time=total_time)
        
        return output_neuron_id



        

#This is just a test

'''
net = Network()

inputs = [2,3,6,1]
input_layer = net.make_group(4,{'spiking':None,'leak':0,'sigmoid':(4,1),'learn_delay':.1})
hidden_layer = net.make_group(4,{'spiking':2,'leak':.1,'sigmoid':(4,1),'learn_delay':.1})
output_layer = net.make_empty_group()

net.add_neuron(neuron_type={'spiking':2,'leak':.1,'sigmoid':(4,1),'learn_delay':.1},
               neuron_group=output_layer,neuron_id=1,is_output=True)

net.add_neuron(neuron_type={'spiking':2,'leak':.1,'sigmoid':(4,1),'learn_delay':.1},
               neuron_group=output_layer,neuron_id=2,is_output=True)

net.connect_groups(input_layer,hidden_layer,.3)

net.connect_groups(hidden_layer,output_layer,.3)

net.input_image(input_layer,inputs,0)

out = net.run(1)
net.plot_neuron_spikes(input_layer[1])

print(out)

'''

