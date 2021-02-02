import numpy as np

########## setting up data ##########
inputs = np.array([[2.0, 4.0, 10.0, 1.0], [2.0, 1.0, 1.0, 1.0], [-1.0, 3.0, 6.0, -2.0]]) # 3 batches 4 inputs/batch
weights = np.array([[1.0, 2.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]) # 2 neurons
biases = 3.0


########## output values of each neuron in the layer with different batches ##########
output = np.dot(inputs, weights.T) + biases
print(output)
