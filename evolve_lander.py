import random
import json
import gym

import numpy as np





"""
LunarLander-v2
Observation , size 8:
min  -1.0169331 -0.4546735 -2.233717  -2.1110673 -4.1701703 -7.3146935   0.         0. 
max   1.0178694  1.7003602  2.3102367  0.58749306 3.7970822  7.351742    1.         1. 


"""




# Save an agent
# Load an agent
# Keep a population of agent (with scores)
# Prune and mutate


# Simple solution. Keep a sorted list where inputs always before their outputs.


class Gate():
    """ Generalized logic gate. Weights are either -1, 0 or 1. Bias is any integer (+0.5).
    Output is dot > bias """
    
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.bias = 0.5
        self.activation = 0
        # Index of the gate in a layer (for book keeping).
        # I use it to decod the output.
        self.index = -1
        # Used for saving circuit into a file.
        self._id = -1

    def serialize(self):
        """ _id is set by the circuit.
        """
        weights = list(self.weights)
        if len(weights) > 0 and not isinstance(weights[0], int):
            weights = [int(w) for w in weights]
        obj = {"weights": weights,
               "inputs": [g._id for g in self.inputs],
               "bias": self.bias,
               "activation": self.activation,
               "index": self.index,
               "id": self._id}
        return obj

    def load_from_json(self, obj, gates):
        """ gates is an array.  Index of gate in the array is the same as gate.index.
        """
        if self._id != obj['id']:
            print("id mismatch")
            return
        self.weights = np.array(obj['weights'])
        self.inputs = [gates[idx] for idx in obj['inputs']]
        self.bias = obj['bias']
        self.activation = obj['activation']
        self.index = obj['index']
        

    def set_id(self, _id):
        self._id = _id

        
    def set_activation(self, value):
        """ Used for input gates"""
        self.activation = value

    def get_activation(self):
        return self.activation

    def set_index(self, value):
        self.index = value

    def get_index(self):
        return self.index
        
    def add_input(self, gate, weight):
        self.weights.append(weight)
        self.inputs.append(gate)

        
    def set_bias(self, bias):
        self.bias = bias

        
    def mutate(self, rate):
        if random.random() < rate:
            self.bias += random.choice([-1,1])
        for idx in range(len(self.weights)):
            if random.random() < rate:
                self.weights[idx] += random.choice([-1,1])

                
    def update(self):
        total = 0
        if len(self.inputs) == 0:
            return
        for x, weight in zip(self.inputs, self.weights):
            total += x.get_activation() * weight
        if total > self.bias:
            self.activation = 1
        else:
            self.activation = 0
            


class Circuit():
    """ Just a graph where each node is a logic gate.
    You have to add gates in directed grpah order.
    """
    def __init__(self):
        self.gates = []

    def assign_gates_ids(self):
        for idx in range(len(self.gates)):
            self.gates[idx].set_id(idx)
        
    def serialize(self):
        self.assign_gates_ids()
        obj = {'gates': [g.serialize() for g in self.gates]}
        return obj

    def load_from_json(self, obj):
        num_gates = len(obj['gates'])
        gates = []
        for idx in range(num_gates):
            gate = Gate()
            self.gates.append(gate)
        self.assign_gates_ids()
        for gate, gate_obj in zip(self.gates, obj['gates']):
            gate.load_from_json(gate_obj, self.gates)
                        
        
    def new_gate(self):
        gate = Gate()
        self.gates.append(gate)
        return gate

    
    def update(self):
        for gate in self.gates:
            gate.update()

            
    def mutate(self, rate):
        """ rate shoudl be from 0 to 1. It is the probability that
        each parameters (weight / bias) will change value.
        values either increment of decrement.
        """
        for gate in self.gates:
            gate.mutate(rate)
    

                
class Agent():
    def __init__(self):
        self.inputs = []
        self.circuit = Circuit()
        self.outputs = []
        # Keep track of all the gates in the last layer for connecting the next.
        self._last_layer_gates = []

        
    def serialize(self):
        obj = {'circuit': self.circuit.serialize(),
               'outputs': [g._id for g in self.outputs],
               'inputs': []}
        for tmp_in in self.inputs:
            in_obj = {'min': tmp_in['min'],
                      'max': tmp_in['max'],
                      'gates': [g._id for g in tmp_in['gates']]}
            obj['inputs'].append(in_obj)
        return obj

    
    def load_from_json(self, obj):
        self.circuit.load_from_json(obj['circuit'])
        gates = self.circuit.gates
        self.inputs = []
        for in_obj in obj['inputs']:
            self.inputs.append({'gates': [gates[idx] for idx in in_obj['gates']],
                                'min': in_obj['min'],
                                'max': in_obj['max']})
        self.outputs = [gates[idx] for idx in obj['outputs']]

        
    def add_input(self, imin, imax, num):
        """ place coding.  Each variable of an observation.
        """
        input_gates = []
        for idx in range(num):
            gate = self.circuit.new_gate()
            self._last_layer_gates.append(gate)
            input_gates.append(gate)
        self.inputs.append({'gates': input_gates,
                            'min': imin,
                            'max': imax})

        
    def set_input_activations(self, observation):
        """ Change an observation vector into place coded input activations.
        Set the input gate activations..
        """
        if len(observation) != len(self.inputs):
            print("Wrong number of input values in observation")
            return
        for in_info, value in zip(self.inputs, observation):
            for gate in in_info['gates']:
                gate.set_activation(0)
            idx = (value - in_info['min']) / (in_info['max'] - in_info['min'])
            idx = int(idx * len(in_info['gates']))
            idx = min(idx, len(in_info['gates'])-1)
            idx = max(idx, 0)
            in_info['gates'][idx].set_activation(1)
            
        
    def add_layer(self, num):
        layer_gates = []
        for idx in range(num):
            gate = self.circuit.new_gate()
            gate.set_index(idx)
            layer_gates.append(gate)
            amin = 0
            amax = 0
            for in_gate in self._last_layer_gates:
                weight = random.randint(-2,2)
                gate.add_input(in_gate, weight)
                if weight < 0:
                    amin += weight
                else:
                    amax += weight
            if amax > amin:
                gate.set_bias(random.randint(amin,amax-1) + 0.5)                
        self._last_layer_gates = layer_gates
        return layer_gates
            
    
    def initialize_lander(self):
        """ Create a random circuit as cart pole agent.
        """
        # Inputs
        self.add_input(-1, 1, 5)
        self.add_input(-4, 1.7, 5)
        self.add_input(-2, 2, 5)
        self.add_input(-2, .5, 5)

        self.add_input(-4, 3.5, 5)
        self.add_input(-7, 7, 5)
        self.add_input(0, 1, 5)
        self.add_input(0, 1, 5)
        # hidden layer
        self.add_layer(7)
        # hidden layer
        self.add_layer(7)
        # outputs
        self.outputs = self.add_layer(4)

        
    def execute_lander(self, observation):
        self.set_input_activations(observation)
        self.circuit.update()
        # Decoding is complicated by the fact that multiple actions can have tru values.
        # Randomly pick one of them.
        active_gates = [g for g in self.outputs if g.activation == 1]
        if len(active_gates) == 0:
            val = random.randint(0,len(self.outputs)-1)
            #val = len(self.outputs)-1
            return val
        action_gate = random.choice(active_gates)
        #action_gate = active_gates[0]
        return action_gate.get_index()

    
    def compute_score(self, env):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            #env.render()
            action = self.execute_lander(observation)
            #print("{}   -> {}".format(observation, action))
            observation, reward, done, info = env.step(action)
            score += reward
        self.score = score
        return score


    def save_to_file(self, filepath):
        with open(filepath, 'w') as fp:
            json.dump(self.serialize(), fp)

            
    def load_from_file(self, filepath):
        with open(filepath, 'r') as fp:
            self.load_from_json(json.load(fp))
           

    def clone(self):
        obj = self.serialize()
        clone = Agent()
        clone.load_from_json(obj)
        return clone

    
    def mutate(self, rate):
        """ rate shoudl be from 0 to 1. It is the probability that
        each parameters (weight / bias) will change value.
        values either increment of decrement.
        """
        self.circuit.mutate(rate)
    



class Population():
    def __init__(self, size):
        self.agents = []
        for idx in range(size):
            agent = Agent()
            agent.initialize_lander()
            self.agents.append(agent)

    def score(self, env):
        best_score = -10000
        for agent in self.agents:
            agent.compute_score(env)
            if agent.score > best_score:
                best_score = agent.score
        return best_score
            
    def prune(self):
        size = len(self.agents)
        self.agents = sorted(self.agents, key=lambda agent: agent.score, reverse=True)
        size = int(size/10)
        self.agents = self.agents[0:size]

    def reproduce(self, mutation_rate):
        new_agents = []
        for agent in self.agents:
            new_agents.append(agent)
            for idx in range(9):
                child = agent.clone()
                child.mutate(mutation_rate)
                new_agents.append(child)
        self.agents = new_agents

    def save_best(self):
        self.agents = sorted(self.agents, key=lambda agent: agent.score, reverse=True)
        agent = self.agents[0]
        agent.save_to_file('lander_%03d.json'%agent.score)

        
        
            

def find_observation_ranges(env):
    observation = env.reset()

    omin = np.array(observation)
    omax = np.array(observation)

    for i in range(1000):
        env.reset()
        done = False
        while not done:
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #pdb.set_trace()
            #print(observation)
            omin = np.minimum(omin, observation)
            omax = np.maximum(omax, observation)

    print(omin)
    print(omax)



def compare_serialized_objects(obj1, obj2):
    if type(obj1) != type(obj2):
        print('type mismatch')
        return False
    if str(type(obj1)) == "<class 'dict'>":
        for key in obj2.keys():
            if not key in obj1:
                print('object1 does not have key %s'%key)
                return
        for key in obj1.keys():
            if not key in obj2:
                print('object2 does not have key %s'%key)
                return
            if not compare_serialized_objects(obj1[key], obj2[key]):
                print("key %s value mismatch"%key)
                return False

    if str(type(obj1)) == "<class 'list'>":
        for v1, v2 in zip(obj1, obj2):
            if not compare_serialized_objects(v1, v1):
                print("array value mismatch")
                return False

    if obj1 != obj2:
        print("value mismatch {} != {}".format(obj1, obj2))
        return False

    return True


def test_serialize_load():
    env = gym.make('CartPole-v1')
    agent1 = Agent()
    agent1.initialize_lander()
    obj1 = agent1.serialize()
    agent2 = Agent()
    agent2.load_from_json(obj1)
    obj2 = agent2.serialize()
    if compare_serialized_objects(obj1, obj2):
        print("success: json is identical")
    env.close()



def test_agent_on_env():
    env = gym.make('LunarLander-v2')
    agent = Agent()
    agent.initialize_lander()
    score = agent.compute_score(env)
    print("success:")
    print("test agent score = %f"%agent.score)
    print("agent has random component, so score may vary.")
    env.close()
    


def test_evolve():
    random.seed(1)
    env = gym.make('LunarLander-v2')
    # This makes 200 networks
    population = Population(200)
    num_generations = 1000
    for idx in range(num_generations):
        best_score = population.score(env)
        print("generation %d, best score %f"%(idx, best_score))
        population.save_best()
        population.prune()
        population.reproduce(0.01)

    env.close()




def test_engineered_agent():
    env = gym.make('LunarLander-v2')
    agent = Agent()

    # Inputs
    agent.add_input(-2.0, 2.0, 3)   # Cart position -2.2 -> 2.4
    agent.add_input(-2.5, 2.5, 3)   # Cart velocity
    agent.add_input(-0.25, 0.25, 2) # pole angle
    agent.add_input(-3, 3, 2)       # tip velocity

    # hidden layer
    gates = agent.add_layer(3)
    gates[0].weights = [0,0,0, 0,0,0, 0,0, 0,1]
    gates[0].bias = 0.5
    gates[1].weights = [0,0,0, 0,0,0, 0,0, 0,0]
    gates[1].bias = 0.5
    gates[2].weights = [0,0,0, 0,0,0, 0,0, 0,0]
    gates[2].bias = 0.5

    # outputs
    gates = agent.add_layer(2)
    agent.outputs = gates
    gates[0].weights = [-1,0,0]
    gates[0].bias = -0.5
    gates[1].weights = [1,0,0]
    gates[1].bias = 0.5

    score = agent.compute_score(env)
    print("success:")
    print("test agent score = %f"%agent.score)
    print("agent has random component, so score may vary.")
    env.close()
    
    
def test_simple_agent():
    env = gym.make('LunarLander-v2')
    agent = Agent()

    observation = env.reset()
    print(observation)
    done = False
    score = 0
    action = 0
    while not done:
        #env.render()
        #action = (action+1)%2
        action = 0

        thresh = 0 #-observation[0] / 50.0

        if observation[3] > thresh:
            action = 1
            
        observation, reward, done, info = env.step(action)
        print("action {}  observation  {}".format(action,observation))
        score += reward

    print(score)



    
    
if __name__ == "__main__":
    #test_agent_on_env()
    #test_serialize_load()
    #test_simple_agent()
    #test_engineered_agent()
    test_evolve()

    #env = gym.make('LunarLander-v2')
    #find_observation_ranges(env)
    




    
