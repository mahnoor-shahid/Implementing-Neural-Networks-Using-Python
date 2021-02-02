import numpy as np

########## setting up data ##########
inputs = np.array([1.0, 2.0])
weights = np.array([[1.0, 2.0], [2.0, 4.0], [1.5, -1.5]]) # 3 neurons in 1 layer
single_bias = 2.0

########## output values of each neuron in the layer ##########
layer_outputs = []

########## estimating summed values produced by each neuron of this layer ##########
for index, neuron_weights in enumerate(weights):
    neuron_output = 0
    for input_value, weight_value in zip(inputs, neuron_weights):
        neuron_output += np.dot(input_value, weight_value)
    neuron_output += single_bias
    print("Neuron {} : {}".format(index+1,neuron_output))
    layer_outputs.append(neuron_output)

print("The final outputs by a single layer of three neurons is: {}" .format(layer_outputs))