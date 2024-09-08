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


bldc = BinaryLogisticDiscriminationClassifier(learningRate=0.01, featureCount=2)


print(f"Training with dataset of size {len(TrainArrayX)} and testing with dataset of size {len(TestArrayX)}")
print(TestArrayY)


for epoch in range(0,3500):
    bldc.learn_step(TrainArrayX, TrainArrayY)

    pred = bldc.predict(TestArrayX)
    # print(f"Predicted output for {len(TestArrayX)} input samples, got predictions for {len(pred)} samples with {np.sum(pred == TestArrayY)} matches.")
    accuracy = np.sum((pred == TestArrayY)) / len(TestArrayY)

    print(f"Epoch {epoch}: accuracy {accuracy}")
