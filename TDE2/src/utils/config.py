import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


class Config:
    class Models:
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
