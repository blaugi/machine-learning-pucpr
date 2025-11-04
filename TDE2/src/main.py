import glob
import os
import pathlib
import zipfile
from utils.config import Config 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from utils.feature_extraction import extract_features

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def extract_zip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def load_and_process_data(relpath: pathlib.Path):

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
    y = pd.read_csv(dataset_path.parent / "processed" / "y_im.csv", header=None)
    y = y.to_numpy()
    y = np.ravel(y)
    print(f"y.shape: {y.shape}")

    # deep features
    X = pd.read_csv(dataset_path.parent / "processed" / "X_im.csv", header=None)
    X = X.to_numpy()
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    Xnew, Xval, ynew, yval = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    return Xnew, Xval, ynew, yval, dataset_path


def main():
    relpath = pathlib.Path(__file__).parent.parent
    Xnew, Xval, ynew, yval, dataset_path = load_and_process_data(relpath)

    # Defining the models and their hyperparameter grids
    models = Config.Models.models
    param_grids = Config.Models.param_grids

    results = []
    best_models = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        grid = param_grids[name]

        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=False,
        )
        gs.fit(Xval, yval)

        best_models[name] = gs.best_estimator_
        print("Melhores hiperparâmetros:", gs.best_params_)

        # Using cross_validation to evaluate the performance
        cv_scores = cross_val_score(gs.best_estimator_, Xnew, ynew, cv=5)
        print("Cross-val (5 folds) média de acurácia:", f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Predictions based on cross_validation
        y_pred = cross_val_predict(gs.best_estimator_, Xnew, ynew, cv=5)

        acc  = accuracy_score(ynew, y_pred)
        prec = precision_score(ynew, y_pred, average="binary")
        rec  = recall_score(ynew, y_pred, average="binary")
        f1   = f1_score(ynew, y_pred, average="binary")

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })


    df_results = pd.DataFrame(results)

    best_name = df_results.loc[df_results["accuracy"].idxmax()]["model"]
    best_estimator = best_models[best_name]

    print(f"Best Model (accuracy): {best_name} — {df_results.loc[df_results['accuracy'].idxmax()]['accuracy']:.4f}")

    y_pred_best = cross_val_predict(best_estimator, Xnew, ynew, cv=5)
    cm = confusion_matrix(ynew, y_pred_best)


    target_names = [folder.split(os.path.sep)[-1] for folder in glob.glob(str(dataset_path  / "Base" / "*"))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    plt.figure(figsize=(5, 5))
    disp.plot(values_format='d', cmap=None)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.savefig(relpath / "figures" / "confusion_matrix_best_model.png")

    print("\nClassification Report (Best model):")
    print(classification_report(ynew, y_pred_best, target_names=target_names))


    plt.figure(figsize=(8, 4))
    plt.bar(df_results['model'], df_results['accuracy'], color='skyblue')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.ylim(0.8, 1.0)  # Set y-axis limit for better visualization of differences
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(relpath / "figures" / "classifier_accuracy_comparison.png")

if __name__ == "__main__":
    main()
