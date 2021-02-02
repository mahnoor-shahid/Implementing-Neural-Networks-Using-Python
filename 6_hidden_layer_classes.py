import numpy as np

########## setting up input ##########
inputs = np.array([[2.0, 4.0, 10.0], [2.0, 1.0, 1.0], [-1.0, 3.0, 6.0]]) 

########## creating hidden layer class ##########
class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = -0.05 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

########## instantiaiting the layers ##########
layer_1 = Layer(3,5)
layer_2 = Layer(5,4)
layer_3 = Layer(4,1)

layer_1.forward(inputs)
layer_2.forward(layer_1.output)
layer_3.forward(layer_2.output)
print(layer_3.output)