import numpy as np

########## setting up data ##########
inputs = np.array([[2.0, 4.0, 10.0, 1.0], [2.0, 1.0, 1.0, 1.0], [-1.0, 3.0, 6.0, -2.0]]) # 3 batches 4 inputs/batch
weights_1 = np.array([[1.0, 2.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]) # 2 neurons layer 1
weights_2 = np.array([[1.0, 4.0], [-1.0, 1.0], [2.0, 1.0]]) # 3 neurons layer 2
weights_3 = np.array([1.0, 1.0, -1.0]) # 1 neuron layer 3
biases = np.array([3.0, 1.0, 2.0])


########## output values of each neuron in respective layer with different batches ##########
layer_1_output = np.dot(inputs, weights_1.T) + biases[0]
layer_2_output = np.dot(layer_1_output, weights_2.T) + biases[1]
layer_3_output = np.dot(layer_2_output, weights_3.T) + biases[2]

print("Layer 1 :  \n" , layer_1_output)
print("Layer 2 :  \n"  , layer_2_output)
print("Layer 3 :  \n" , layer_3_output)

########## Can loop over layers ##########
for index, item in enumerate(biases):
    print("#######################")
    print("Layer " + str(index+1))
    if index == 0:
        weights = weights_1
    elif index == 1:
        inputs = output
        weights = weights_2
    else:
        inputs = output
        weights = weights_3

    output = np.dot(inputs, weights.T) + biases[index]
    print(" Outputs : \n" + str(output))


