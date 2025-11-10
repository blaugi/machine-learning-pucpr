import cv2
import numpy as np
import pandas as pd
import os
import glob

import pathlib
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input

def extract_features():
    W = 224
    H = 224
    model = InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_tensor=Input(shape=(W, H, 3)),
    )

    relpath = pathlib.Path(__file__).parent.parent.parent
    print(relpath)
    file_dict= {}
    path = relpath / "data" / "raw" / "Base"
    for folder in glob.glob(str(path / "*")):
        file_dict[folder] = os.listdir(folder)
    

    X_deep = []
    y = []
    paths = []  # List to store file paths

    for (class_folder, instance_files), classe in zip(file_dict.items(), range(len(file_dict))):
        for i in range(len(instance_files)):
            file_name = (
                str(os.path.join(class_folder, instance_files[i]))
            )
            y.append(classe)
            paths.append(file_name)  # Append file path
            
            imagem = cv2.imread(file_name)
            img = cv2.resize(imagem, (W, H))
            xd = image.img_to_array(img)
            xd = np.expand_dims(xd, axis=0)
            xd = preprocess_input(xd)
            X_deep.append(xd)

    X_deep = np.asarray(X_deep)
    X_deep = X_deep.reshape(X_deep.shape[0], W, H, 3)
    X = model.predict(X_deep)

    # Salva as características extraídas em um csv (um vetor de valores para cada imagem)
    df = pd.DataFrame(X)
    csv_base = relpath / "data" / "processed"
    if not os.path.exists(csv_base):
        os.makedirs(csv_base)
    print(f"Saving features to {csv_base / 'X_im.csv'}")
    df.to_csv(csv_base / "X_im.csv", header=False, index=False)

    # Salva y que contém a classe de cada imagem
    df_class = pd.DataFrame(y)
    print(f"Saving labels to {csv_base  / 'y_im.csv'}")
    df_class.to_csv(csv_base/ "y_im.csv", header=False, index=False)

    df_paths = pd.DataFrame(paths)
    print(f"Saving paths to {csv_base / 'paths_im.csv'}")
    df_paths.to_csv(csv_base / "paths_im.csv", header=False, index=False)
