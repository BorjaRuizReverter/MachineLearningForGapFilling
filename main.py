#main.py

import numpy as np

class NeuralNetwork():  #create the one and only class we'll need
   
    def __init__(self):
        np.random.seed(1)  #seed the random number generator

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1  #make your own custom 3 x 1 matrix and assign random weights to it with values ranging from -1 to 1 and 0 mean

    def sigmoid(self, x): #takes in each individually weighted input's sum of the inputs and transforms them between 0 and 1 through the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):  #the derivative of the sigmoid function used to calculate necessary weight adjustments
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):  #We train the model through trial and error, adjusting the synaptic weights each time to get a better result

        for iteration in range(training_iterations):
            output = self.think(training_inputs)  #pass training set through the neural network
            error = training_outputs - output  #calculate the error rate
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))  #multiply error by input and gradient of the sigmoid function, less confident weights are adjusted more through the nature of the function
            self.synaptic_weights += adjustments  #adjust synaptic weights

    def think(self, inputs):  #pass inputs through the neural network to get output
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":  #initialize the single neuron neural network
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],  #the training set, with 4 examples consisting of 3 input values and 1 output value 
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_inputs, training_outputs, 10000)  #train the neural network
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: ")) #these are the three inputs it will take, as show in the table above
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
   
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))