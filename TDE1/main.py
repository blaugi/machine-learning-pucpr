from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy import ndarray
import random



random.seed(42)

def predict_knn(train_X: ndarray, train_y: ndarray, test_X: ndarray, k: int) -> ndarray:
    n_test = test_X.shape[0]
    
    test_X_exp = test_X[:, np.newaxis, :] 
    train_X_exp = train_X[np.newaxis, :, :]
    diff = test_X_exp - train_X_exp
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    k_indices = np.argpartition(dist, k, axis=1)[:, :k]
    # argpartition is really weird so ill leave an explanation that made sense here:
    # there is first a partition step, in which it places the kth value in a new array and does this:
    #  [lower value elements, kth element, higher value elements]
    # then argpartition changes the elements for their indices in the original array 
    # np.array([9, 2, 7, 4, 6, 3, 8, 1, 5])
    # np.argpartition(array, 5)
    # would result in: array([5, 8, 7, 3, 1, 4, 6, 2, 0])
    
    predictions = []
    for i in range(n_test):
        neighbors_labels = train_y[k_indices[i]]
        pred = np.bincount(neighbors_labels.astype(int)).argmax()
        predictions.append(pred)
    return np.array(predictions)

def calculate_accuracy(pred: ndarray, true: ndarray) -> float:
    return np.mean(pred == true) * 100



def main():
    dataset_choice = 'breast'

    digits = load_digits()
    breast = load_breast_cancer()

    if dataset_choice == 'digits':
        X = digits.data
        y = digits.target
        dataset_name = 'Digits'
    elif dataset_choice == 'breast':
        X = breast.data
        y = breast.target
        dataset_name = 'Breast Cancer'
    else:
        raise ValueError("Invalid dataset_choice. Choose 'digits' or 'breast'.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Using {dataset_name} dataset")

    pred_manual = predict_knn(X_train, y_train, X_test, k=5)
    acc_manual = calculate_accuracy(pred_manual, y_test)
    print(f"Manual KNN Accuracy: {acc_manual:.2f}%")
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, y_train)
    pred_sklearn = knn_sklearn.predict(X_test)
    acc_sklearn = calculate_accuracy(pred_sklearn, y_test)
    print(f"Sklearn KNN Accuracy: {acc_sklearn:.2f}%")

    





if __name__ == "__main__":
    main()
