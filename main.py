from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = pd.read_csv("./data/prepared_data/diagnosis.csv", encoding='UTF-16')

print(data)