# -*- coding: utf-8 -*-
# Import Statements
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# %% Retrieve IMU Data
# Data Format -> GyroX, GyroY, GyroZ, AccX, AccY, AccZ

def process_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        tmp = lines[i].split(",")
        for j in range(len(tmp)):
            tmp[j] = float(tmp[j])
        lines[i] = tmp
    
    return np.array(lines)

test_idle1 = process_data("test_idle1.txt")
test_idle2 = process_data("test_idle2.txt")
test_idle3 = process_data("test_idle3.txt")
test_nonidle1 = process_data("test_nonidle1.txt")
test_nonidle2 = process_data("test_nonidle2.txt")
test_nonidle3 = process_data("test_nonidle3.txt")
upward_lift1 = process_data("upward_lift1.txt")
upward_lift2 = process_data("upward_lift2.txt")
upward_lift3 = process_data("upward_lift3.txt")
forward_push1 = process_data("forward_push1.txt")
forward_push2 = process_data("forward_push2.txt")
forward_push3 = process_data("forward_push3.txt")
rotation1 = process_data("rotation1.txt")
rotation2 = process_data("rotation2.txt")
rotation3 = process_data("rotation3.txt")

# %% Test Cell
arr = np.reshape(np.arange(9), (3, 3))
tester = forward_push3
# tester = test_idle3
print(tester.mean(axis=0, keepdims=True))
print(tester.max(axis=0, keepdims=True))

# %% Part (2) -> Idle v. Non-idle
threshold = 0.25 # Idle threshold

gestures = [0, 1] # 0 -> Idle, 1 -> Nonidle
train_idle_data_x = [test_idle1, test_idle2, test_idle3,
                   test_nonidle1, test_nonidle2, test_nonidle3]
train_idle_data_y = [0, 0, 0, 1, 1, 1]
train_idle_data_pred = []


for data in train_idle_data_x:
    # Compare AccX to see if we're idle or not
    val = data.max(axis=0, keepdims=True)[0, 3]
    
    pred = 1 if (val > threshold) else 0
    train_idle_data_pred.append(pred)
    pass

print("Comparing pred v. truth for idle classifications!")
print(train_idle_data_y)
print(train_idle_data_pred)
print("Error:", str(np.mean(np.array(train_idle_data_y) - np.array(train_idle_data_pred))))

# %% Part (3) -> Differentiate between (push, lift) & idle
clf = DecisionTreeClassifier(random_state=0)
gestures_part3 = [0, 1, 2] # 0 -> Idle, 1 -> Forward push, 2 -> Upward lift
X = np.array([test_idle1, test_idle2, test_idle3,
              forward_push1, forward_push2, forward_push3,
              upward_lift1, upward_lift2, upward_lift3])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
clf.fit(np.reshape(X, (X.shape[0], -1)), y)

y = clf.predict(np.reshape(X, (X.shape[0], -1)))

# %% Retrieve Testing Data
# N.B. Testing is stored
x1 = process_data("testing-idle.txt")
x2 = process_data("testing-push.txt")
x3 = process_data("testing-push1.txt")
x4 = process_data("testing-lift.txt")
x5 = process_data("testing-lift1.txt")
X_test = np.array([x1, x2, x3, x4, x5])
X_test = np.reshape(X_test, (X_test.shape[0], -1))
y_test = [0, 1, 1, 2, 2]
# Decision Tree Classifier has 40% accuracy
# [idle, push, push, lift, lift]
# [idle, idle, push, push, idle]
y = clf.predict(X_test)
err = 0
num_trials = 5
for i in range(num_trials):
    if y[i] != y_test[i]:
        err += 1
err /= num_trials
print("With three actions our error:", str(err))
# %% Retrieve Rotation Values
x6 = process_data("testing-rotation1.txt")
x7 = process_data("testing-rotation2.txt")

# %% Part (4) -> Differentiate between 4 actions (adding rotation)
X_train = np.array([test_idle1, test_idle2, test_idle3,
              forward_push1, forward_push2, forward_push3,
              upward_lift1, upward_lift2, upward_lift3,
              rotation1, rotation2, rotation3])
y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]) # 3 -> Rotation
clf.fit(np.reshape(X_train, (X_train.shape[0], -1)), y_train)
y_train_pred = clf.predict(np.reshape(X_train, (X_train.shape[0], -1)))

# %% Testing from the classifier
X_test1 = np.array([x1, x2, x3, x4, x5, x6, x7])
X_test1 = np.reshape(X_test1, (X_test1.shape[0], -1))
y_test = np.array([0, 1, 1, 2, 2, 3, 3])

y_test_pred = clf.predict(X_test1)

final_err = 0
num_trials1 = y_test.shape[0]

for i in range(num_trials1):
    if (y_test_pred[i] != y_test[i]):
        final_err += 1
final_err /= num_trials1

print("With 4 actions, our error is:", str(final_err))

'''
Yes we saw gravity acceleration when idle, as we displayed ~1G when converting
the raw output to G units.

Some values drift when idle, at least when using the base given code. This is 
seen in the behavior of the gyroscope. A good feature for idle v. non-idle is 
to look at the accelerometer values, as x and y should be close to 0 if idle.
At an idle threshold of 0.25, we recorded 100% accuracy. All 3 idles are 
correctly classified as idle, and all 3 non-idle are correctly classified as
non-idle. 


The simple classifier simply used the time series of all gyroscope rates and 
accelerometer values. I used decision tree to classify structurally.



Yes you can use the same features, and arguably increased our accuracy as we
were exposed to rotation motions, which helped distinctly separate the upwards
push and lift motions more clearly via the Decision Tree. Circular rotation motions
were tracked perfectly on the test set. Our features simply involved a 1D time 
series array of all of our accelerometer and gyroscope rate values (6 vals). 
Actions that might be easier include changing our classifier, maybe incorporating
some sort of measurement on largest deviation between inputs, mean, max, etc.
'''



# %%
num_gestures = 3 # Tune this
imu_data = None # Training data
test_data = None # Testing data
y = None # Actual testing data labels
model = KMeans(n_clusters=num_gestures,
               init="k-means++",
               random_state=0)
# model.fit(imu_data) # Train the classifier
# test_labels = model.predict(test_data)

# %% Gather Accuracies

