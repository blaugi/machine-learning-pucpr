from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import ndarray
import random



random.seed(42)


def np_find_dist(s_array: ndarray) -> ndarray:
    n = len(s_array)  
    a = s_array.reshape(n, 1)
    b = s_array.reshape(1, n)
    dist = np.sum((a - b)**2, axis=-1)
    return dist

def np_k_nearest(dist: ndarray, k: int) -> ndarray:
    k_indices = np.argpartition(dist, k+1, axis=1)[:, :k+1]
    return k_indices

def np_main(points:ndarray, count: int = 6):
    k = 2
    array = points
    np_dist = np_find_dist(array)
    k_indices = np_k_nearest(np_dist, k)

    results = [array[k_indices[i, :k+1]] 
               for i in range(array.shape[0])]
    return results, array, k_indices, k


def main():

    digits = load_digits()
    breast = load_breast_cancer()

    X_digits:ndarray = digits.data
    y_digits:ndarray= digits.target   
    
    X_breast:ndarray= breast.data
    y_breast:ndarray = breast.target 

    digits_X_train, digits_X_test, digits_y_train, digits_y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)


    digits_train = np.array(list(zip(digits_X_train, digits_y_train)), dtype=object)
    # X_train, X_test, y_train, y_test = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)

    np_find_dist(digits_X_train)  # Call on features only





if __name__ == "__main__":
    main()
