import sys
import numpy as np
import matplotlib

########## versions of python and packages ##########
print('Python: {} '.format(sys.version))
print('Numpy: {} ' .format(np.__version__))
print ('Matplotlib: {}'.format (matplotlib.__version__))

########## coding single neuron ##########
inputs = [1.1, 2.1, 1.4]
weights = [0.2, 1.4, -0.5]
bias = 3 

# computing the output of the single neuron
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
output = output + bias
print("The neuron will give " , output)


