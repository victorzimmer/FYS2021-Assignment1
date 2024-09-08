# Import NumPy, Pandas, Datetime, OS, and Matplotlib.pyplot
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

# Import custom LogisticRegression class
from BinaryLogisticDiscriminationClassifier import BinaryLogisticDiscriminationClassifier

# Create folder for outputs
RUN_NAME = str(datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
RUN_PATH = "./output/"+RUN_NAME+"/"
os.mkdir(RUN_PATH)

# Load dataset
SpotifyDataset = pd.read_csv("data/SpotifyFeatures.csv")

# Separate Pop and Classical into datasets
PopDataset = SpotifyDataset.loc[SpotifyDataset["genre"] == "Pop"]
ClassicalDataset = SpotifyDataset.loc[SpotifyDataset["genre"] == "Classical"]
print(f"Dataset contains {len(PopDataset)} songs in the Pop genre and {len(ClassicalDataset)} songs in the Classical genre.")

# Set label column in both datasets
PopDataset = PopDataset.assign(label=1.0)
ClassicalDataset = ClassicalDataset.assign(label=0.0)

# Drop all unused columns from both datasets
neededColumns = ["label", "liveness", "loudness"]
PopDataset = PopDataset.drop([x for x in list(PopDataset.columns) if (x not in neededColumns)], axis=1)
ClassicalDataset = ClassicalDataset.drop([x for x in list(ClassicalDataset.columns) if (x not in neededColumns)], axis=1)

# Calculate split point for 80/20 distribution of both classes
splitPointPop = int(len(PopDataset) * 0.8)
splitPointClassical = int(len(ClassicalDataset) * 0.8)

# Create train and test datasets by concatenating pop and classical based on split point
TrainDataset = pd.concat([PopDataset.iloc[0:splitPointPop], ClassicalDataset.iloc[0:splitPointClassical]])
TestDataset = pd.concat([PopDataset.iloc[splitPointPop:len(PopDataset)], ClassicalDataset.iloc[splitPointClassical:len(ClassicalDataset)]])

if False:
    print(TestDataset)
    print(TrainDataset)

# Convert training dataset to numpy arrays for X and Y
TrainArrayX = np.array(TrainDataset.iloc[:, 0:2])
TrainArrayY = np.array(TrainDataset.iloc[:, 2:3]).T[0]

# Convert testing dataset to numpy arrays for X and Y
TestArrayX = np.array(TestDataset.iloc[:, 0:2])
TestArrayY = np.array(TestDataset.iloc[:, 2:3]).T[0]

if False:
    print(TrainArrayX, TrainArrayY)
    print(TestArrayX, TestArrayY)

# Scatter both distributions (before split) on a plot
plt.scatter(list(PopDataset.iloc[:, 0]), list(PopDataset.iloc[:, 1]), c="#0A014F", alpha=0.5)
plt.scatter(list(ClassicalDataset.iloc[:, 0]), list(ClassicalDataset.iloc[:, 1]), c="#CD9FCC", alpha=0.5)

# Set the names for the legend
plt.legend(["Pop", "Classical"])

# Name the labels based on the column names from the dataset
plt.xlabel(PopDataset.columns[0])
plt.ylabel(PopDataset.columns[1])

# Save the figure to the folder created for this run
plt.savefig(RUN_PATH+"liveness-loudness-plot.png")

if False:
    plt.show()
plt.clf()

# Create new instance of BLDC
bldc = BinaryLogisticDiscriminationClassifier(learningRate=0.01, featureCount=2)


print(f"Training with dataset of size {len(TrainArrayX)} and testing with dataset of size {len(TestArrayX)}")
print(TestArrayY)

# Set epoch range
epochs = range(0,1000)
# Collect training errors for plotting
trainingErrors = []

# Training loop
for epoch in epochs:
    # Learn a step
    trainingError = bldc.learn_step(TrainArrayX, TrainArrayY)
    # Store error for epoch
    trainingErrors.append(trainingError)

    # Get predictions for accuracy computation
    trainPred = bldc.predict(TrainArrayX)
    testPred = bldc.predict(TestArrayX)

    # Compute accuracies
    trainAccuracy = np.sum((trainPred == TrainArrayY)) / len(TrainArrayY)
    testAccuracy = np.sum((testPred == TestArrayY)) / len(TestArrayY)

    print(f"Epoch {epoch}: train-accuracy {trainAccuracy}, test-accuracy {testAccuracy}")



# Get predictions on test set
testPred = bldc.predict(TestArrayX)
# Count true/false positive/negative for confusion matrix
truePositives = np.sum(((testPred == 1) & (TestArrayY == 1)))
falsePositives = np.sum(((testPred == 1) & (TestArrayY == 0)))
falseNegatives = np.sum(((testPred == 0) & (TestArrayY == 1)))
trueNegatives = np.sum(((testPred == 0) & (TestArrayY == 0)))
print(f"{len(testPred)} total predictions. {truePositives} TP, {falsePositives} FP, {falseNegatives} FN, {truePositives} TN")

# Choose random false positive and negative
indexOfRandomFalseNegative = np.random.choice(np.where((testPred == 0) & (TestArrayY == 1))[0])
indexOfRandomFalsePositive = np.random.choice(np.where((testPred == 1) & (TestArrayY == 0))[0])
trackNameFalsePositive = "SONG"
trackNameFalseNegative= "SONG"

# Find the corresponding songs in TestDataset
falsePositiveRow = TestDataset.iloc[indexOfRandomFalsePositive]
falseNegativeRow = TestDataset.iloc[indexOfRandomFalseNegative]
# Extract their liveness and loudness
livenessFalsePositive = falsePositiveRow["liveness"]
loudnessFalsePositive = falsePositiveRow["loudness"]
livenessFalseNegative = falseNegativeRow["liveness"]
loudnessFalseNegative = falseNegativeRow["loudness"]

# Find the corresponding track in the orignal SpotifyDataset
matching_tracks_fp = SpotifyDataset[
    (SpotifyDataset["liveness"] == livenessFalsePositive) &
    (SpotifyDataset["loudness"] == loudnessFalsePositive)
]

matching_tracks_fn = SpotifyDataset[
    (SpotifyDataset["liveness"] == livenessFalseNegative) &
    (SpotifyDataset["loudness"] == loudnessFalseNegative)
]


# Get trackand artist name for fale postive and negative
trackNameFalsePositive = matching_tracks_fp.iloc[0]["track_name"] + " by " + matching_tracks_fp.iloc[0]["artist_name"]
trackNameFalseNegative = matching_tracks_fn.iloc[0]["track_name"] + " by " + matching_tracks_fn.iloc[0]["artist_name"]

print(f"False positive at {indexOfRandomFalsePositive} which is the song {trackNameFalsePositive}, a Classical song that was identified as Pop")
print(f"False positive at {indexOfRandomFalseNegative} which is the song {trackNameFalseNegative}, a Pop song that was identified as Classical")

# Plot trainig error over epochs
plt.plot(epochs, trainingErrors)
plt.xlabel("Epoch")
plt.ylabel("Training Error")

# Save the figure to the folder created for this run
plt.savefig(RUN_PATH+"trainingError-epoch-plot.png")

if False:
    plt.show()
plt.clf()



# Scatter both distributions (before split) on a plot
plt.scatter(list(PopDataset.iloc[:, 0]), list(PopDataset.iloc[:, 1]), c="#0A014F", alpha=0.5)
plt.scatter(list(ClassicalDataset.iloc[:, 0]), list(ClassicalDataset.iloc[:, 1]), c="#CD9FCC", alpha=0.5)

# Extract learned parameters from the model [bias, liveness, loudness]
bias, liveness, loudness = bldc.parameters

# Find range of existing plotted data
print(PopDataset.iloc[:, 0].min(), PopDataset.iloc[:, 0].max())

# Compute values for the decision boundary
x_values = np.linspace(PopDataset.iloc[:, 0].min(), PopDataset.iloc[:, 0].max(), 100)
y_values = -(bias + liveness * x_values) / loudness

# Plot the decision boundary
plt.plot(x_values, y_values, color='red', label="Decision Boundary")

# Set the names for the legend
plt.legend(["Pop", "Classical", "Decision Boundary"])

# Name the labels based on the column names from the dataset
plt.xlabel(PopDataset.columns[0])
plt.ylabel(PopDataset.columns[1])

# Save the figure to the folder created for this run
plt.savefig(RUN_PATH+"decision-boundary-plot.png")

if False:
    plt.show()
plt.clf()
