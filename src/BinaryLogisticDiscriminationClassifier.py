# Import numpy
import numpy as np
import random

# Simoid helper function
def sigmoid(z):
    expZ = np.exp(np.clip(z, -500,500))
    return expZ / (1 + expZ)

class BinaryLogisticDiscriminationClassifier:
    # featureCount: Number of features and thereby parameters, not including bias
    # learningRate: Learning rate
    def __init__(self, featureCount=1, learningRate=0.01, stochasticSelectionSize=10, optimizer="SGD"):
        # Parameters for the model, with parameter 0 being the bias and always added
        self.parameters = np.zeros(featureCount+1)

        # Learning rate, the "speed" at which we follow the stochastic gradient descent
        self.learningRate = learningRate

        # Size of sample for stochastic gradient descent
        self.stochasticSelectionSize = stochasticSelectionSize

        # Verify and store optimizer selection
        if not (optimizer in ["SGD", "GD"]):
            print("Error: Selected optimizer that has not been implemented!")
        self.optimizer = optimizer



    # X: NDArray[samples, features], contains input variables for each feature, for each sample
    # y: NDArray[samples], contains labels for each sample, 0 or 1
    # Returns training error
    def learn_step(self, X, y):
        # Add column of ones to input for bias term
        X = np.hstack([np.ones((len(X), 1)), X])

        # Compute linear prediction(s)
        linear_prediction = np.dot(X, self.parameters)

        # Apply sigmoid function to get logistic prediction(s)
        logistic_prediction = sigmoid(linear_prediction)

        # Calculate training error (Binary Cross-Entropy Loss)
        training_error = -(1/len(X)) * np.sum(y * np.log(logistic_prediction) + (1 - y) * np.log(1 - logistic_prediction))

        # Gradient descent
        if self.optimizer == "GD":
            parameter_gradient = (1/len(X)) * np.dot(X.T, (logistic_prediction - y))

        # Stochastic gradient descent
        elif self.optimizer == "SGD":
            stochasticSelection = random.choice(range(len(X)-self.stochasticSelectionSize))
            parameter_gradient = (1/self.stochasticSelectionSize) * np.dot(X[stochasticSelection:stochasticSelection+self.stochasticSelectionSize].T, (logistic_prediction[stochasticSelection:stochasticSelection+self.stochasticSelectionSize] - y[stochasticSelection:stochasticSelection+self.stochasticSelectionSize]))

        else:
            print(f"Error: Optimizer not found: {self.optimizer}.")
            return

        self.parameters = self.parameters - (self.learningRate * parameter_gradient)

        return training_error


    def predict(self, X):
        # Add column of ones to input for bias term
        X = np.hstack([np.ones((len(X), 1)), X])

        # Compute linear prediction(s)
        linear_prediction = np.dot(X, self.parameters)

        # Apply sigmoid function to get logistic prediction(s)
        logistic_prediction = sigmoid(linear_prediction)

        # Return class predictions, either 1 or 0
        return np.array([1 if y > 0.5 else 0 for y in logistic_prediction])
