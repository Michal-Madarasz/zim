import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

if __name__ == '__main__':
    data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')
    print(data)

    predictors = ["Temperature", "Nausea", "LumberPain", "ConUrine",
                  "MictPains", "UrethraBurn"]

    selector = SelectKBest(f_classif, k=5)
    # TODO How to handle text data in scikit-learn
    # selector.fit(data[predictors], data["Inflammation"])
    # scores = -np.log10(selector.pvalues_)
