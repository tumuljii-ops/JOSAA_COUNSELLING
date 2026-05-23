import numpy as np
import pandas as pd

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("josaa_final_dataset.csv")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Institute average closing rank
df["inst_avg_rank"] = df.groupby(

    "Institute"

)["Closing Rank"].transform("mean")

# Program average closing rank
df["prog_avg_rank"] = df.groupby(

    "Program"

)["Closing Rank"].transform("mean")

# Institute + Program average
df["inst_prog_avg"] = df.groupby(

    ["Institute", "Program"]

)["Closing Rank"].transform("mean")

# =====================================================
# TARGET
# =====================================================

Y = df["Closing Rank"]

# =====================================================
# FEATURES
# =====================================================

X = df.drop(columns=["Closing Rank"])

# =====================================================
# ONE HOT ENCODING
# =====================================================

X = pd.get_dummies(X)

print("\nShape After Encoding:", X.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(

    X,
    Y,

    test_size=0.2,

    random_state=42,

    shuffle=True
)

# =====================================================
# IMPORTS
# =====================================================

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# =====================================================
# KFOLD
# =====================================================

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)

# =====================================================
# HYPERPARAMETER SEARCH SPACE
# =====================================================

learning_rates = [0.01]

max_depths = [4]

n_estimators_list = [500]

subsamples = [0.7]

colsample_values = [0.7]

min_child_weights = [1]

gammas = [0]

reg_alphas = [0]

reg_lambdas = [3]

# =====================================================
# BEST VARIABLES
# =====================================================

best_score = -float("inf")

best_params = None

# =====================================================
# GRID SEARCH + KFOLD
# =====================================================

for lr in learning_rates:

    for depth in max_depths:

        for estimators in n_estimators_list:

            for subsample in subsamples:

                for colsample in colsample_values:

                    for child_weight in min_child_weights:

                        for gamma in gammas:

                            for alpha in reg_alphas:

                                for reg_lambda in reg_lambdas:

                                    fold_scores = []

                                    print("\n====================================")

                                    print(

                                        f"LR={lr}, "
                                        f"Depth={depth}, "
                                        f"Estimators={estimators}, "
                                        f"Subsample={subsample}, "
                                        f"Colsample={colsample}, "
                                        f"ChildWeight={child_weight}, "
                                        f"Gamma={gamma}, "
                                        f"Alpha={alpha}, "
                                        f"Lambda={reg_lambda}"
                                    )

                                    # ================================
                                    # KFOLD LOOP
                                    # ================================

                                    for train_idx, val_idx in kf.split(X_train):

                                        X_fold_train = X_train.iloc[train_idx]

                                        X_fold_val = X_train.iloc[val_idx]

                                        y_fold_train = Y_train.iloc[train_idx]

                                        y_fold_val = Y_train.iloc[val_idx]

                                        # ============================
                                        # MODEL
                                        # ============================

                                        model = XGBRegressor(

                                            learning_rate=lr,

                                            max_depth=depth,

                                            n_estimators=estimators,

                                            subsample=subsample,

                                            colsample_bytree=colsample,

                                            min_child_weight=child_weight,

                                            gamma=gamma,

                                            reg_alpha=alpha,

                                            reg_lambda=reg_lambda,

                                            objective='reg:squarederror',

                                            random_state=42,

                                            n_jobs=-1
                                        )

                                        # ============================
                                        # TRAIN
                                        # ============================

                                        model.fit(

                                            X_fold_train,

                                            y_fold_train
                                        )

                                        # ============================
                                        # PREDICTIONS
                                        # ============================

                                        val_pred = model.predict(

                                            X_fold_val
                                        )

                                        # ============================
                                        # R2 SCORE
                                        # ============================

                                        fold_r2 = r2_score(

                                            y_fold_val,

                                            val_pred
                                        )

                                        fold_scores.append(

                                            fold_r2
                                        )

                                    # ================================
                                    # AVERAGE SCORE
                                    # ================================

                                    avg_score = np.mean(

                                        fold_scores
                                    )

                                    print(

                                        "Average CV R²:",

                                        avg_score
                                    )

                                    # ================================
                                    # STORE BEST
                                    # ================================

                                    if avg_score > best_score:

                                        best_score = avg_score

                                        best_params = {

                                            "learning_rate": lr,

                                            "max_depth": depth,

                                            "n_estimators": estimators,

                                            "subsample": subsample,

                                            "colsample_bytree": colsample,

                                            "min_child_weight": child_weight,

                                            "gamma": gamma,

                                            "reg_alpha": alpha,

                                            "reg_lambda": reg_lambda
                                        }

# =====================================================
# BEST PARAMETERS
# =====================================================

print("\n====================================")
print("BEST PARAMETERS")
print("====================================")

print(best_params)

print("\nBest CV R²:", best_score)

# =====================================================
# FINAL MODEL
# =====================================================

final_model = XGBRegressor(

    **best_params,

    objective='reg:squarederror',

    random_state=42,

    n_jobs=-1
)

# =====================================================
# TRAIN FINAL MODEL
# =====================================================

final_model.fit(

    X_train,

    Y_train
)

# =====================================================
# TEST PREDICTIONS
# =====================================================

test_pred = final_model.predict(

    X_test
)

# =====================================================
# FINAL METRICS
# =====================================================

final_r2 = r2_score(

    Y_test,

    test_pred
)

final_rmse = np.sqrt(

    mean_squared_error(

        Y_test,

        test_pred
    )
)

# =====================================================
# FINAL RESULTS
# =====================================================

print("\n====================================")
print("FINAL TEST RESULTS")
print("====================================")

print("Final Test R²:", final_r2)

print("Final RMSE:", final_rmse)

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

results = pd.DataFrame({

    "Actual Rank": Y_test[:10],

    "Predicted Rank": test_pred[:10]
})

print("\n====================================")
print("SAMPLE PREDICTIONS")
print("====================================")

print(results)

#saving model
import joblib

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(

    final_model,

    "model.pkl"
)

print("Model saved successfully")

# ============================================
# SAVE TRAINING COLUMNS
# ============================================

joblib.dump(

    X_train.columns.tolist(),

    "columns.pkl"
)

print("Columns saved successfully")

# ============================================
# SAVE REFERENCE DATASET
# ============================================

reference_df = df[[
    "Institute",
    "Program",
    "Quota",
    "Seat Type",
    "Gender",
    "College_Type",
    "Opening Rank"
]].copy()

reference_df = reference_df.drop_duplicates()

joblib.dump(

    reference_df,

    "reference_df.pkl"
)

print("Reference dataset saved successfully")