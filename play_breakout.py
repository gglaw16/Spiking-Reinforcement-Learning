# -*- coding: utf-8 -*-
"""
This should use a spiking neural network to play breakout (the atari game)
"""

import gym
import numpy as np
from PIL import Image


def sigmoid(X):
   return 1/(1+np.exp(-X))


"""
Object that contains the network
calculates outputs of layers
"""
class Network(object):
    
    def __init__(self, num_hidden_layers, layer_sizes, input_size, lr):
        self.learning_rate = lr
        self.layers = []
        self.layers.append(Layer(layer_sizes[0],(input_size+
                                 layer_sizes[0]+layer_sizes[1])))
        
        for i in range(1,num_hidden_layers):
            self.layers.append(Layer(layer_sizes[i],(layer_sizes[i-1]+
                                     layer_sizes[i]+layer_sizes[i+1])))
            
        self.layers.append(Layer(layer_sizes[num_hidden_layers],
                                             layer_sizes[num_hidden_layers-1]
                                             +layer_sizes[num_hidden_layers]))
        
        
    # this just runs through the network normally    
    def run_forward(self,image,action):
        last_output = self.calculate_layer_output(0,image)
        for i in range(1,len(self.layers)-1):
            last_output = self.calculate_layer_output(i,last_output)
        
        self.set_network_output(action)
        
    def dream(self):
        back1 = np.matmul(np.array([0,1,0,0]),np.transpose(self.layers[1].get_weights()))
        back1 /= back1.max()

        first_out = back1[:200]
        
        image = np.matmul(first_out,np.transpose(self.layers[0].get_weights()))
        image /= image.max()
        image = image[:100800]
        
        return image*255

        
        
    def calculate_layer_output(self,layer,prev_layer_input,dreaming=False):
        weights = self.layers[layer].get_weights()
        inputs = np.concatenate((prev_layer_input,
                                 self.layers[layer].get_last_output(),
                                 self.layers[layer+1].get_last_output()))
        
        layer_output = np.matmul(inputs,weights)
        layer_output /= layer_output.max()
        layer_output = sigmoid(layer_output)
        
        layer_output = 1.0*(layer_output > np.random.random(layer_output.shape))
        
        if not dreaming:
            self.calculate_weights(layer,inputs,layer_output)
        
        self.layers[layer].set_last_output(layer_output)
        
        return layer_output
        
    def set_network_output(self,network_output):
        layer = len(self.layers)-1
        inputs = np.concatenate((self.layers[layer-1].get_last_output(),
                                 self.layers[layer].get_last_output()))
        
        self.calculate_weights(layer,inputs,network_output)
        
        self.layers[layer].set_last_output(network_output)
        
    def calc_network_output(self, dreaming=False):
        layer = len(self.layers)-1
        weights = self.layers[layer].get_weights()
        inputs = np.concatenate((self.layers[layer-1].get_last_output(),
                                 self.layers[layer].get_last_output()))
        
        layer_output = sigmoid(np.matmul(inputs,weights))
        
        layer_output = 1.0*(layer_output > np.random.random(layer_output.shape))
        
        if not dreaming:
            self.calculate_weights(layer,inputs,layer_output)
        
        self.layers[layer].set_last_output(layer_output)
        
        return layer_output
        
    def calculate_weights(self,layer,inputs,outputs):
        weight_change = np.outer(inputs,outputs)*self.learning_rate
        new_weights = self.layers[layer].get_weights() + weight_change
        new_weights = new_weights / new_weights.max(axis=0)
        self.layers[layer].set_weights(new_weights)
        
        
"""
Object that contains a layer
layer should have weights
should keep the last output
"""
class Layer(object):
    
    def __init__(self, size, inputs_size):
        self.size = size
        self.weights = np.random.rand(inputs_size,size)
        self.last_output = np.zeros(size)
  
    def get_weights(self):
        return self.weights
    
    def set_weights(self, new_weights):
        self.weights = new_weights
    
    def get_last_output(self):
        return self.last_output
    
    def set_last_output(self, new_output):
        self.last_output = new_output
        
        
    
if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    image_size = 210*160*3
    network = Network(1,np.array([200,4]),image_size,.01)
    
    for i_episode in range(1):
        print("Episode %d"%i_episode)
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            
            image = observation.reshape(210*160*3)
            image = image/image.max()
            output_vector = np.zeros(4)
            output_vector[action] = 1
            network.run_forward(image, output_vector)
            
            observation, reward, done, info = env.step(action)
            print(action)

    env.close()
    
    dreamed_image = network.dream()
    
    dreamed_image = dreamed_image.reshape((210,160,3))
    img = Image.fromarray(dreamed_image, 'RGB')
    img.show()
    
    
    