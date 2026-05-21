import numpy as np
import pandas as pd

df=pd.read_csv("josaa_final_dataset.csv")

#spliting data into X(features) AND Y(Target)

Y=df["Closing Rank"]
X=df.drop(columns=["Closing Rank"])

print("\n input features shape")
print(X.shape)

print("\n output feature shape is")
print(Y.shape)

#-----------ONE HOT ENCODING------------------
X=pd.get_dummies(X)


print("Shape of X is:",X.shape)

print("First 5 rows are:",X.head())

import numpy as np
import pandas as pd

df=pd.read_csv("josaa_final_dataset.csv")

#spliting data into X(features) AND Y(Target)

Y=df["Closing Rank"]
X=df.drop(columns=["Closing Rank"])

print("\n input features shape")
print(X.shape)

print("\n output feature shape is")
print(Y.shape)

#-----------ONE HOT ENCODING------------------
X=pd.get_dummies(X)


print("Shape of X is:",X.shape)

print("First 5 rows are:",X.head())

# =====================================================
# TRAIN TEST SPLIT USING SKLEARN
# =====================================================

from sklearn.model_selection import train_test_split

# Convert dataframe to numpy arrays
X = X.values.astype(float)
Y = Y.values.astype(float)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\n===================================")
print("TRAIN TEST SPLIT COMPLETED")
print("===================================")

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTraining Target Shape:")
print(Y_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTesting Target Shape:")
print(Y_test.shape)

#standarization of the data

mean=X_train.mean(axis=0)  #axis=0 means col wise

print("mean shape is",mean.shape)

std=X_train.std(axis=0)

print("std shape is:",std.shape)

std[std==0]=1

X_train=(X_train-mean)/std

X_test=(X_test-mean)/std

print("The standarisation resutl is",X_train[0][:10])


# STANDARDIZE TARGET VARIABLE (Y)


# Compute mean and std of target
Y_mean = Y_train.mean()

Y_std = Y_train.std()

# Prevent division by zero
if Y_std == 0:
    Y_std = 1

# Standardize training target
Y_train = (Y_train - Y_mean) / Y_std

# Standardize testing target
Y_test = (Y_test - Y_mean) / Y_std

print("\nTarget Standardization Completed!")

print("\nSample Y values:")
print(Y_train[:10])

# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_model(learning_rate, epochs):

    # Number of features
    n_features = X_train.shape[1]

    # Initialize weights
    weights = np.zeros(n_features)

    # Initialize bias
    bias = 0

    # Number of samples
    n_samples = X_train.shape[0]

    # Store losses
    losses = []


    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(epochs):

        # Forward propagation
        predictions = np.dot(X_train, weights) + bias


        # MSE Loss
        mse = np.mean(
            (Y_train - predictions) ** 2
        )

        losses.append(mse)


        # Gradients
        dw = (-2 / n_samples) * np.dot(
            X_train.T,
            (Y_train - predictions)
        )

        db = (-2 / n_samples) * np.sum(
            (Y_train - predictions)
        )


        # Update parameters
        weights = weights - learning_rate * dw

        bias = bias - learning_rate * db


    return losses

#---------------hyper_parameter tuning --------------

learning_rates=[0.1,0.01,0.001,0.0001]
epoch_values=[100,500,1000,5000]


best_loss = float("inf")

best_lr = None

best_epochs = None


for lr in learning_rates:

    for epochs in epoch_values:

        print(
            f"\nTraining with LR={lr}, Epochs={epochs}"
        )

        losses = train_model(
            learning_rate=lr,
            epochs=epochs
        )

        final_loss = losses[-1]

        print(
            f"Final Loss: {final_loss}"
        )


        # Store best result
        if final_loss < best_loss:

            best_loss = final_loss

            best_lr = lr

            best_epochs = epochs


print("\n===================================")
print("BEST HYPERPARAMETERS")
print("===================================")

print(f"Best Learning Rate: {best_lr}")

print(f"Best Epochs: {best_epochs}")

print(f"Best Loss: {best_loss}")
         
         
    
    
    
    




