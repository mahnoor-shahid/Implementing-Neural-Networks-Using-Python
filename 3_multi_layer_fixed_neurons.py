import numpy as np

########## setting up data ##########
inputs = np.array([1.0, 2.0])
weights_1 = np.array([[1.0, 2.0], [2.0, 4.0], [1.0,1.0]])
weights_2 = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
weights_3 = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]])
biases = [2.0, 1.0, 2.0]
n_layers = 3


########## output values of each neuron in the layer 1 and 2 ##########
layer_outputs = []


########## estimating summed values produced by each neuron of each layer ##########
for index_bias in range(n_layers):

    # selecting the layer
    if(index_bias==0):
        weights = weights_1
    elif(index_bias==1):
        weights = weights_2
        inputs = layer_outputs
    else:
        weights = weights_3
        inputs = layer_outputs
    
    # reseting the outputs
    layer_outputs = []

    print("######## Iteration {} ########". format(index_bias+1))
    print("Inputs are ", inputs)
    print("Weights associated with each input is ", weights)

    # looping over that particular layer
    for index, neuron_weights in enumerate(weights):
        neuron_output = 0
        for input_value, weight_value in zip(inputs, neuron_weights):
            neuron_output += np.dot(input_value, weight_value)
        neuron_output += biases[index_bias]
        print("Neuron {} : {}".format(index+1,neuron_output))
        layer_outputs.append(neuron_output)
    print("The layer {} output is {}".format(index_bias+1, layer_outputs))

