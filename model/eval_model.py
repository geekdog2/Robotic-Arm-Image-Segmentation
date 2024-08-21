#!/usr/bin/env python3



import numpy as np
from sklearn.model_selection import KFold
from train_model import GraspingPoseEval

# Assuming you have your data and labels stored in X and y respectively

# Define the number of folds for cross-validation
num_folds = 5

# Initialize a list to store the performance metrics for each fold
performance_metrics = []
input_size = 10
hidden_size = 10
out_size = 1

model=GraspingPoseEval(input_size, hidden_size, output_size)

# Create a KFold object
kf = KFold(n_splits=num_folds)

# Perform cross-validation
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train your model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    performance = model.evaluate(X_test, y_test)

    # Store the performance metric for the current fold
    performance_metrics.append(performance)

# Calculate the average performance metric across all folds
average_performance = np.mean(performance_metrics)

# Print the average performance metric
print("Average performance:", average_performance)