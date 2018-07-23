#This is the siimplest neural network implementation
#Its the point from where my neural network journey begins

import numpy as np

#using sigmoid activation function
def activate(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#toy data
input_data = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
output_labels = np.array([[0], [0], [1], [1]])

synaptic_weight_0 = 2*np.random.random((3,4)) - 1
synaptic_weight_1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

	# Forward propagate through layers 0, 1, and 2
    layer0 = input_data
    layer1 = activate(np.dot(layer0,synaptic_weight_0))
    layer2 = activate(np.dot(layer1,synaptic_weight_1))

    #calculate error for layer 2
    layer2_error = output_labels - layer2

    if (j% 10000) == 0:
        print ("Error:", str(np.mean(np.abs(layer2_error))))

    #Use it to compute the gradient
    layer2_gradient = layer2_error*activate(layer2,deriv=True)

    #calculate error for layer 1
    layer1_error = layer2_gradient.dot(synaptic_weight_1.T)

    #Use it to compute its gradient
    layer1_gradient = layer1_error * activate(layer1,deriv=True)

    #update the weights using the gradients
    synaptic_weight_1 += layer1.T.dot(layer2_gradient)
    synaptic_weight_0 += layer0.T.dot(layer1_gradient)


#testing
print(activate(np.dot(np.array([1, 0, 0]), synaptic_weight_0)))
