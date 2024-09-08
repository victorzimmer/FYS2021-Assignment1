#set document(
  title: "FYS-2021 Machine Learning, Assignment 1",
  author: "Victor Zimmer <vzi002@uit.no>",
  date: auto
)

#import "template/uit-assignment.typ": conf
#show: doc => conf(
  faculty: "Faculty of Science and Technology",
  title: "Assignment 1",
  subtitle: "Logistic Regression",
  name: "Victor Zimmer",
  course: "FYS-2021 Machine Learning",
  semester: "Autumn",
  year: "2024",
  doc,
)

= Introduction
In this assignment we will read data from a CSV file and train a computer on it. The data consists of Spotify tracks and our goal is to use the liveness and loudness features of the tracks to classify
them in the genres "Pop" and "Classical". To do this we will use logistic regression.

The assignment consists of preprocessing data, implementing a logistic dicrimination classifier, training the classifier, and evaluating the resulting model.

= Problem 1: Preprocessing the data
The data needed for logistic regression is separated into inputs with features as columns and samples as rows, and labels for their expected outputs with a single column containing the label for each sample along the rows.

== Problem 1a
To load the dataset we use the Pandas library and its method for reading CSV files
```Python
# Load dataset
SpotifyDataset = pd.read_csv("data/SpotifyFeatures.csv")
```

== Problem 1b
We want to extract only the rows in the dataset belonging to the genres "Pop" and "Classical". The dataset contains 9386 songs in the Pop genre and 9256 songs in the Classical genre.
We also want to reduce the dataset to only contain the features we're interested in ("liveness" & "loudness") as well as a label which should be 1 for Pop songs and 0 for Classical songs.
```Python
# Separate Pop and Classical into datasets
PopDataset = SpotifyDataset.loc[SpotifyDataset["genre"] == "Pop"]
ClassicalDataset = SpotifyDataset.loc[SpotifyDataset["genre"] == "Classical"]

# Set label column in both datasets
PopDataset = PopDataset.assign(label=1.0)
ClassicalDataset = ClassicalDataset.assign(label=0.0)

# Drop all unused columns from both datasets
neededColumns = ["label", "liveness", "loudness"]
PopDataset = PopDataset.drop([x for x in list(PopDataset.columns) if (x not in neededColumns)], axis=1)
ClassicalDataset = ClassicalDataset.drop([x for x in list(ClassicalDataset.columns) if (x not in neededColumns)], axis=1)
```

== Problem 1c
Finally to get the data ready for training and testing we'll convert them to NumPy arrays and split them 80% / 20% between training and testing sets respectively, such that we end up with
four final arrays, x values for test and train, and y values for test and train. In the x values features will be columns and samples will be rows. The y values have a single column, the label, and one row for each sample.

```Python
# Calculate split point for 80/20 distribution of both classes
splitPointPop = int(len(PopDataset) * 0.8)
splitPointClassical = int(len(ClassicalDataset) * 0.8)

# Create train and test datasets by concatenating pop and classical based on split point
TrainDataset = pd.concat([PopDataset.iloc[0:splitPointPop], ClassicalDataset.iloc[0:splitPointClassical]])
TestDataset = pd.concat([PopDataset.iloc[splitPointPop:len(PopDataset)], ClassicalDataset.iloc[splitPointClassical:len(ClassicalDataset)]])

# Convert training dataset to numpy arrays for X and Y
TrainArrayX = np.array(TrainDataset.iloc[:, 0:2])
TrainArrayY = np.array(TrainDataset.iloc[:, 2:3])

# Convert testing dataset to numpy arrays for X and Y
TestArrayX = np.array(TestDataset.iloc[:, 0:2])
TestArrayY = np.array(TestDataset.iloc[:, 2:3])
```

== Problem 1d
We will plot the liveness vs. loudness of the dataset using a scatterplot, with a different color for each class.
#image("images/liveness-loudness-plot.png")
From the plot is is appearent that there is a difference between the classes where a possible decision boundary could be, however it isn't likely to be 100% precise as there
are some samples in each class that have overlapping values for both features.

= Problem 2: Teaching a machine
Now we get to the actual machine learning and will implement a logistic regression discriminator, which will fit a function to the data to classify samples between Pop and Classical based on the loudness and liveness.

To implement a logistic discrimination classifier we will first need to define it mathematically. Logistic discrimination classifiers are often referred to as logistic regression, even tough they are classifiers, not regressions.
We also assume it is sufficient to implement a binary classifier as we only have two classes, Pop and Classical.

== Problem 2a
We should implement a logistic discrimination classifier, using stochastic gradient descent for optimization. It should be implemented with the learning rate as a hyperparameter and
in a manner where it is possible to report the error as a function of epochs during training.

Our classifier will learn the function $g(bold(upright(x))) in [0,1]$, which will return a value estimating the class of the input data $bold(upright(x))$.

== Linear Regression
We begin with definitions from linear regression where the goal is to fit a linear function $y="ax"+b$ to best predict the output given the training data.
For a single sample it looks like $f(bold(upright(x))) = y = bold(beta)_0 + bold(upright(x))_1 bold(beta)_1 + ... + bold(upright(x))_f bold(beta)_f + epsilon$ for every $f$ feature in the input.
where $bold(upright(x))$ is a vector containing the features of the input and $bold(beta)$ is a vector of the parameters/weights. $bold(beta)_0$ is the bias, corresponding to the $b$ in the linear function.
Finally epsilon is included to represent the error.

With a single sample the linear regression prediction can then be calculated as $f(bold(upright(x))) = y = 1 bold(beta)_0 + bold(upright(x))_1 bold(beta)_1 + ... + bold(upright(x))_f bold(beta)_f + epsilon = bold(beta) bold(upright(x))$, assuming the first value of $bold(upright(x))$ is always $1$.

For multiple samples linear regression can efficiently be computed using matrices as $f(bold(upright(X))) = bold(upright(y)) = bold(upright(X)) bold(beta) + bold(epsilon)$ where every sample becomes $y_i = bold(upright(x))_"i0" bold(beta)_"0" + bold(upright(x))_"i1" bold(beta)_1 + ... + bold(upright(x))_"if" bold(beta)_f + epsilon_i$
with an assumption that $bold(upright(x))_"i0" = 1$ for all samples $i$.

Optimizing linear regression is done by measureing the loss using the mean squared error (MSE), which is simply the mean of all euclidean distances from
each point to the line squared. This gives use the loss function $"MSE"(bold(hat(y)) = f(bold(upright(x))), bold(y)) = sum (hat(y)_i - y_i)^2$, and thus fitting the linear regression
becomes a problem of minimizing this loss. This has a closed-form solution that is not relevant for this assignment, but suffice to say it would be an easy task to compute.

== Binary Logistic Discrimination Classifier
To go from linear regression to a binary logistic discrimination classifier we use the logistic function to change the $y$ values of a
linear regression from $[-inf,inf]$ to $[0,1]$, this function is $s(z) = frac(e^z, 1+e^z)$ and if usually referred to as the sigmoid function.

We apply the sigmoid function to our equation for linear regression $f(x)=bold(beta)^T bold(upright(x))$, to get a function for the prediction using logistic regression $g(x)=frac(e^(bold(beta)^T bold(upright(x))), 1+e^(bold(beta)^T bold(upright(x))))$.

This also extends for multiple samples as $g(bold(upright(X))) = frac(e^(bold(upright(X)) bold(beta)), 1+e^(bold(upright(X)) bold(beta)))$.

We have omitted the error term, which we do with an assumption that the error has mean value of $epsilon=0$, allowing the bias
term to account for the entire mean error.

Optimizing logistic regression we can no longer use the MSE loss from linear regression as it assumes a linear relationship between the predicted and true values.


== Stochastic Gradient Descent vs Gradient Descent
This implementation should use stochastic gradient descent, as opposed to gradient descent.
The difference lies in the samples used to compute the gradient, in the case of gradient descent the entire dataset is used,
whilst stochastic gradient descent uses random samples from the dataset re-selected every iteration.

Stochastic gradient descent can therefore deal with much larger datasets, whilst gradient descent will usually give a smoother path to the minima.


== Implementation in Python
The implementation has a class _LogisticDiscriminationClassifier_ that holds the logic for learning parameters and making predictions.
It assumes the training loop is handled outside the class, to enable reporting per-epoch data.
```Python
class BinaryLogisticDiscriminationClassifier:
    def __init__(self, featureCount, learningRate):

    def learn_step(self, X, y):

    def predict(self, X):
```


=== Sigmoid helper function
We will need to use the sigmoid function multiple times, so to avoid writing it out every time we implement a helper function for computing it.
The implementation is based on the function $s(z)=frac(e^z,1+e^z)$ from above. For more efficient computation we avoid computing the exponential $e^z$ twice.
```Python
# Simoid helper function
def sigmoid(z):
    expZ = np.exp(z)
    return expZ / (1 + expZ)
```

=== Prediction
First we implement the prediction as it follows from our $g(bold(upright(X)))$ above. We also need to make sure our returned values
follow the expectation of being either 0 or 1, by returning 1 for values above 0.5 and 0 otherwise.
```Python
def predict(self, X):
    # Add column of ones to input for bias term
    X = np.hstack([np.ones((len(X), 1)), X])

    # Compute linear prediction(s)
    linear_prediction = np.dot(X, self.parameters)

    # Apply sigmoid function to get logistic prediction(s)
    logistic_prediction = sigmoid(linear_prediction)

    # Return class predictions, either 1 or 0
    return [1 if y > 0.5 else 0 for y in logistic_prediction]
```


=== Learning
Learning the logistic regression is done by computing the gradient for the loss with respect to the input.
Then the parameters are adjusted slightly, based on the learning rate, in that direction.
This is repeated for any number of steps requested. In this implementation the learning function will carry out a single step,
relying on an external training loop to direct it.

==== Stochastic Gradient Descent
Both gradient descent and stochastic gradient descent (SGD) are implemented, with SGD being the default.
It also default to using 10 samples for SGD, with the option of changing the value.
```Python
# Gradient descent
parameter_gradient = (1/len(X)) * np.dot(X.T, (logistic_prediction - y))

# Stochastic Gradient Descent
stochasticSelection = random.choice(range(len(X)-self.stochasticSelectionSize))
parameter_gradient = (1/self.stochasticSelectionSize) * np.dot(X[stochasticSelection:stochasticSelection+self.stochasticSelectionSize].T, (logistic_prediction[stochasticSelection:stochasticSelection+self.stochasticSelectionSize] - y[stochasticSelection:stochasticSelection+self.stochasticSelectionSize]))
```
