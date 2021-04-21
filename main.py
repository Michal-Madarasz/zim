import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split

from names_dict import COLUMNS

if __name__ == '__main__':
    data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')

    predictors = dict(COLUMNS[1:])

    selector = SelectKBest(f_classif, k=3)
    for i in predictors:
        data[i] = data[i].map({'yes': 1, 'no': 0})

    print(data)

    num_of_neighbours = [1, 5, 10]
    type_of_metric = ["euclidean", "manhattan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=.30,
        random_state=1234
    )

    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
    print(rkf)
    scores = []

