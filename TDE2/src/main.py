import glob
import os
import pathlib
import zipfile
from utils.config import Config
import altair as alt
import datetime
import mlflow

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

experiment_name = f"TDE2_Model_Comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
mlflow.set_experiment(experiment_name)


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

    paths = pd.read_csv(dataset_path.parent / "processed" / "paths_im.csv", header=None)
    paths = paths.to_numpy().ravel()
    print(f"paths.shape: {paths.shape}")

    Xnew, Xval, ynew, yval = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    indices = np.arange(len(X))
    indices_new, indices_val = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    paths_new = paths[indices_new]
    paths_val = paths[indices_val]
    return Xnew, Xval, ynew, yval, dataset_path, paths_new


def main():
    relpath = pathlib.Path(__file__).parent.parent
    Xnew, Xval, ynew, yval, dataset_path, paths = load_and_process_data(relpath)
    print(f"Xnew.shape: {Xnew.shape}, Xval.shape: {Xval.shape}, ynew.shape: {ynew.shape}, yval.shape: {yval.shape}")

    # Defining the models and their hyperparameter grids
    models = Config.Models.models
    param_grids = Config.Models.param_grids

    results = []
    best_models = {}
    best_run_id = None
    run_ids_by_model = {}  

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_complete_evaluation") as run:
            current_run_id = run.info.run_id
            run_ids_by_model[name] = current_run_id  # Store run ID for this model
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

            mlflow.set_tags({"model_type": name, "phase": "complete_evaluation"})
            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("best_cv_score", gs.best_score_)

            best_models[name] = gs.best_estimator_
            print("Melhores hiperparâmetros:", gs.best_params_)

            # Using cross_validation to evaluate the performance
            cv_scores = cross_val_score(gs.best_estimator_, Xnew, ynew, cv=5)
            print(
                "Cross-val (5 folds) média de acurácia:",
                f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})",
            )

            # Log cross-validation metrics
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())

            # Predictions based on cross_validation
            y_pred = cross_val_predict(gs.best_estimator_, Xnew, ynew, cv=5)

            acc = accuracy_score(ynew, y_pred)
            prec = precision_score(ynew, y_pred, average="weighted")
            rec = recall_score(ynew, y_pred, average="weighted")
            f1 = f1_score(ynew, y_pred, average="weighted")

            # Log final evaluation metrics
            mlflow.log_metrics({
                "final_accuracy": acc,
                "final_precision": prec,
                "final_recall": rec,
                "final_f1": f1
            })

            results.append(
                {
                    "model": name,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )

    df_results = pd.DataFrame(results)

    best_name = df_results.loc[df_results["accuracy"].idxmax()]["model"]
    best_estimator = best_models[best_name]
    best_run_id = run_ids_by_model[best_name]  # Get run ID for best model

    print(
        f"Best Model (accuracy): {best_name} — {df_results.loc[df_results['accuracy'].idxmax()]['accuracy']:.4f}"
    )

    y_pred_best = cross_val_predict(best_estimator, Xnew, ynew, cv=5)
    cm = confusion_matrix(ynew, y_pred_best)

    target_names = [
        folder.split(os.path.sep)[-1]
        for folder in glob.glob(str(dataset_path / "Base" / "*"))
    ]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    plt.figure(figsize=(5, 5))
    disp.plot(values_format="d", cmap=None)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    plt.savefig(relpath / "figures" / "confusion_matrix_best_model.png")

    # Log confusion matrix to best model's run
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(str(relpath / "figures" / "confusion_matrix_best_model.png"))

    print("\nClassification Report (Best model):")
    print(classification_report(ynew, y_pred_best, target_names=target_names))

    misclassified_indices = np.where(y_pred_best != ynew)[0]
    print(f"\nNumber of misclassified samples: {len(misclassified_indices)}")
    
    if len(misclassified_indices) > 0:
        os.makedirs(relpath / "results", exist_ok=True)
        misclassified_df = pd.DataFrame({
            'index': misclassified_indices,
            'true_label': [target_names[y] for y in ynew[misclassified_indices]],
            'predicted_label': [target_names[y] for y in y_pred_best[misclassified_indices]],
            'file_path': paths[misclassified_indices]  # Add file paths
        })
        misclassified_df.to_csv(relpath / "results" / "misclassified_samples.csv", index=False)
        print(f"Misclassified samples saved to {relpath / 'results' / 'misclassified_samples.csv'}")

    plt.figure(figsize=(8, 4))
    plt.bar(df_results["model"], df_results["accuracy"], color="skyblue")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy Comparison")
    plt.ylim(0.8, 1.0)  # Set y-axis limit for better visualization of differences
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(relpath / "figures" / "classifier_accuracy_comparison.png")

    bar = (
        alt.Chart(df_results)
        .mark_bar()
        .encode(
            x="model",
            y="accuracy",
        )
        .properties(
            width=alt.Step(40)  # controls width of bar.
        )
    )
    bar.save(relpath / "figures" / "classifier_accuracy_comparison.html")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(str(relpath / "figures" / "classifier_accuracy_comparison.png"))
        mlflow.log_artifact(str(relpath / "figures" / "classifier_accuracy_comparison.html"))


if __name__ == "__main__":
    main()
