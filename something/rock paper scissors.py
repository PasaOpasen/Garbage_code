
import random
import numpy as np




def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)



# Dense layer
class Layer:
    def __init__(self, from_size, to_size, activation):
        
        self.from_size = from_size
        self.to = to_size
        self.activation = activation
        
        self.weights_total = (from_size + 1) * to_size
    
    def set_weights(self, vector, from_index):
        to_index = from_index + self.weights_total
        array = vector[from_index:to_index]
        
        self.M = array.reshape((self.to, self.from_size + 1))
        
        return to_index
        
    def evaluate(self, input_):
        v = self.M[:,:-1].dot(input_[:, np.newaxis]).flatten() + self.M[:, -1]
        return self.activation(v)



# Dense neural network ended by 3 units with softmax
class NN:
    def __init__(self, flatten_weights, input_size, hidden_sizes = [], hidden_activations = []):
        
        layers = []
        
        if len(hidden_sizes) == 0:
            layers.append(Layer(input_size, 3, softmax))
            
            def eval(input_):
                answer = layers[0].evaluate(input_)
                return int(np.argmax(answer))
            
        else:
            layers.append(Layer(input_size, hidden_sizes[0], hidden_activations[0]))
            for i in range(1, len(hidden_sizes)):
                layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], hidden_activations[i]))
            layers.append(Layer(hidden_sizes[-1], 3, softmax))
            
            def eval(input_):
                answer = layers[0].evaluate(input_)
                for layer in layers[1:]:
                    answer = layer.evaluate(answer)
                return int(np.argmax(answer))
        
        self.evaluate = eval   
        
        # read weights from flatten vector
        
        ind = layers[0].set_weights(flatten_weights, 0)
        for layer in layers[1:]:
            ind = layer.set_weights(flatten_weights, ind)



win = lambda answer: int((answer + 1) % 3)

observed_rounds = 10
input_size = observed_rounds * 3

#nn = NN(np.loadtxt('best_weights.txt'), input_size, [], [])
nn = NN(np.zeros(93), input_size, [], [])
last_ans = 0


def agent(observation, configuration):
    global last_ans 
        
    data = np.empty(input_size, dtype = np.int8)
        
    if observation.step == 0:
        answer = random.randint(0, 2)
        data[0] = answer
        last_ans = answer
        
    elif observation.step < observed_rounds:
        answer = random.randint(0, 2)
        i = observation.step*3
        last_ans = answer
        data[i] = answer 
        data[i-2] = observation.lastOpponentAction
        data[i-1] = get_score(data[i-3], data[i-2])
        
    elif observation.step == observed_rounds:
        i = observation.step*3
        data[i-2] = observation.lastOpponentAction
        data[i-1] = get_score(data[i-3], data[i-2])
            
        answer = nn.evaluate(data)
        last_ans = answer
            
    else: # we can start to observe, and shift observations after each game
        data[:-3] = data[3:]
        data[-3] = last_ans
        data[-2] = observation.lastOpponentAction
        data[-1] = get_score(data[-3], data[-2])
            
        answer = nn.evaluate(data)
        last_ans = answer
        
    return win(answer)



