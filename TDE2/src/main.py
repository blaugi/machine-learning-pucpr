import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pathlib

from utils.feature_extraction import extract_features


def extract_zip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def main():
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    relpath = pathlib.Path(__file__).parent.parent

    zip_path = relpath / "data" / "raw" / "Base.zip"
    dataset_path = relpath / "data" / "raw"
    if os.path.exists(os.path.join(dataset_path, "Base")):
        print("Dataset already exists. Skipping extraction.")
        extract_zip(zip_path, dataset_path)
    else:
        print(f"Extracting dataset from {zip_path} to {dataset_path}.")
        extract_zip(zip_path, dataset_path)
    if os.path.exists(zip_path / "_MACOSX"):
        print("Removing _MACOSX directory.")
        os.rmdir(zip_path / "_MACOSX")

    # check the file for better info on what this does
    extract_features()

    # Labels
    y = pd.read_csv("y_im.csv", header=None)
    y = y.to_numpy()
    y = np.ravel(y)
    print(y.shape)
    # deep features
    X = pd.read_csv("X_im.csv", header=None)
    X = X.to_numpy()
    print(X.shape)
    print(y.shape)

    # TODO Change this to kfold cross validation
    Xnew, Xval, ynew, yval = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    (
        X.shape,
        Xnew.shape,
        Xval.shape,
        ynew.value_counts(normalize=True),
        yval.value_counts(normalize=True),
    )

    # Defining the models and their hyperparameter grids
    models = {
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "RF": RandomForestClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(random_state=RANDOM_STATE),
        "Xgboost": xgb.XGBClassifier(
            objective="binary:logistic", random_state=RANDOM_STATE
        ),
    }


    param_grids = {
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "DecisionTree": {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "NaiveBayes": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(20,), (15,), (15, 10), (50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3, 1e-2],
        },
        "SVM": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear", "poly"],
            "clf__gamma": ["scale", "auto"],
        },
        "RF": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],
        },
        "Bagging": {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.5, 0.75, 1.0],
        },
        "Xgboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],
        },
    }


if __name__ == "__main__":
    main()
