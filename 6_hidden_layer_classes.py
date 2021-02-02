import numpy as np

inputs = np.array([[2.0, 4.0, 10.0], [2.0, 1.0, 1.0], [-1.0, 3.0, 6.0]]) 

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.05 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer(3,5)
layer2 = Layer(5,4)

layer1.forward(inputs)
layer2.forward(layer1.output)
print(layer2.output)