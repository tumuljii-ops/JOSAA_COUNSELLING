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

    return weights, bias, losses


 # =====================================================
 # HYPERPARAMETER TUNING
 # =====================================================

learning_rates = [0.1, 0.01, 0.001, 0.0001]

epoch_values = [100, 500, 1000, 5000]

best_loss = float("inf")

best_lr = None

best_epochs = None

best_weights = None

best_bias = None

best_losses = None


for lr in learning_rates:

    for epochs in epoch_values:

        print(f"\nTraining with LR={lr}, Epochs={epochs}")

        weights, bias, losses = train_model(
            learning_rate=lr,
            epochs=epochs
        )

        final_loss = losses[-1]

        print(f"Final Loss: {final_loss}")

        # Detect exploding loss
        if np.isnan(final_loss) or np.isinf(final_loss):

            print("Loss exploded. Skipping...")

            continue

        # Store best result
        if final_loss < best_loss:

            best_loss = final_loss

            best_lr = lr

            best_epochs = epochs

            best_weights = weights

            best_bias = bias

            best_losses = losses
            
 #convergence analysis
            
            
import matplotlib.pyplot as plt

plt.plot(best_losses)

plt.xlabel("Epochs")

plt.ylabel("MSE Loss")

plt.title("Loss Convergence Curve")

plt.show()

#final model summary


print("\n===================================")
print("BEST HYPERPARAMETERS")
print("===================================")

print(f"Best Learning Rate: {best_lr}")

print(f"Best Epochs: {best_epochs}")

print(f"Best Loss: {best_loss}")



 # =====================================================
 # TEST PREDICTIONS
 # =====================================================

Y_pred = np.dot(X_test, best_weights) + best_bias


# =====================================================
# TEST MSE
# =====================================================

test_mse = np.mean(
    (Y_test - Y_pred) ** 2
)

#-----------RMSE SCORE--------------------

print("\nTest MSE:", test_mse)


rmse = np.sqrt(test_mse)

print("RMSE:", rmse)

#--------------R**2 SCORE ------------------
ss_res = np.sum(
    (Y_test - Y_pred) ** 2
)

ss_tot = np.sum(
    (Y_test - np.mean(Y_test)) ** 2
)

r2_score = 1 - (ss_res / ss_tot)

print("R2 Score:", r2_score)

# =====================================================
# INVERSE TRANSFORM
# =====================================================

Y_pred_original = (Y_pred * Y_std) + Y_mean

Y_test_original = (Y_test * Y_std) + Y_mean


         
    
    
    
    




