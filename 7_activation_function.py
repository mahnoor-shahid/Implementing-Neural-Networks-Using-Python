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


########## creating activation layer class ##########
class ReLu:    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


########## instantiaiting the layers ##########
layer_1 = Layer(3,5)
relu_1 = ReLu()
layer_2 = Layer(5,4)
relu_2 = ReLu()
layer_3 = Layer(4,1)

layer_1.forward(inputs)
print("Layer 1 : \n" , layer_1.output)
relu_1.forward(layer_1.output)
print("Layer 1 Relu : \n" ,relu_1.output)
layer_2.forward(relu_1.output)
print("Layer 2 : \n" ,layer_2.output)
relu_2.forward(layer_2.output)
print("Layer 2 Relu : \n" ,relu_2.output)
layer_3.forward(layer_2.output)
print("Layer 3 : \n" ,layer_3.output)