from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

from names_dict import FEATURES

if __name__ == '__main__':
    data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')

    predictors = dict(FEATURES)
    predict_no_temperature = dict(FEATURES[1:])

    selector = SelectKBest(f_classif, k=3)
    for i in predict_no_temperature:
        data[i] = data[i].map({'yes': 1, 'no': 0})

    selector.fit_transform(data[predictors], data["Inflammation"])
    scores = -np.log10(selector.pvalues_)

    predictors_list = list(predictors.values())
    result = []
    for i in range(len(scores)):
        result.append(tuple((scores[i], predictors_list[i])))

    sorted(result, key=itemgetter(0))

    result.sort()

    plt.figure(figsize=(12, 6))
    plt.axes([0.45, 0.35, 0.5, 0.5])
    plt.barh(range(len(result)), dict(result).keys())
    plt.yticks(range(len(result)), dict(result).values())
    plt.show()
