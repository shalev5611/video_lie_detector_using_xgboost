import os
import pandas as pd
import csv
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import f1_score, precision_score, recall_score

import pandas as pd
import csv
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import f1_score, precision_score, recall_score

model = xgboost.XGBClassifier()
rows = ['frame', ' face_id', ' timestamp', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
au = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
ex_au = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']


def most_frequent(List):
    return max(set(List), key = List.count)

def average(List):
    return sum(List)/len(List)


X = []
Y = []

df = pd.read_csv("truesight_csv.csv")

def search_dataframe(d,element):
    for i in d.iterrows():
        if str(i[1][0]) == element:
            return i[1][1]


with os.scandir("au_data/processed/") as it:
    for entry in it:
        if entry.name.endswith(".csv") and entry.is_file():
            l = search_dataframe(df,str(entry.name))
            Y.append(l)
            en = []

            csv = pd.read_csv(entry.path)
            row = {}
            for i in rows:
                row[i] = []
                for j in csv[i]:
                    row[i].append(j)

            for i in au:
                en.append(most_frequent(row[i]))
            for i in ex_au:
                en.append(average(row[i]))

            X.append(en)

X = np.array(X)
Y = np.array(Y)
print(X.shape,Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, stratify = Y)


model.fit(X_train,y_train)
model.load_model("model.json")

y_pred = model.predict(X_test)

print(model.predict(np.array([X_test[12]])))

positive = 0
overall = len(y_test)
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        positive +=1
model.save_model("model.json")
print("accuracy score", positive/overall)
print("precision score", precision_score(y_test,y_pred))
print("recall score", recall_score(y_test,y_pred))
print("f1 score", f1_score(y_pred,y_test))
rows = ['frame', ' face_id', ' timestamp', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
au = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
ex_au = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']


def most_frequent(List):
    return max(set(List), key = List.count)

def average(List):
    return sum(List)/len(List)


X = []
Y = []

df = pd.read_csv("truesight_csv.csv")

def search_dataframe(d,element):
    for i in d.iterrows():
        if str(i[1][0]) == element:
            return i[1][1]


with os.scandir("au_data/processed/") as it:
    for entry in it:
        if entry.name.endswith(".csv") and entry.is_file():
            l = search_dataframe(df,str(entry.name))
            Y.append(l)
            en = []

            csv = pd.read_csv(entry.path)
            row = {}
            for i in rows:
                row[i] = []
                for j in csv[i]:
                    row[i].append(j)

            for i in au:
                en.append(most_frequent(row[i]))
            for i in ex_au:
                en.append(average(row[i]))

            X.append(en)

X = np.array(X)
Y = np.array(Y)
print(X.shape,Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, stratify = Y)


model.fit(X_train,y_train)
model.load_model("model.json")

y_pred = model.predict(X_test)

print(model.predict(np.array([X_test[12]])))

positive = 0
overall = len(y_test)
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        positive +=1
model.save_model("model.json")
print("accuracy score", positive/overall)
print("precision score", precision_score(y_test,y_pred))
print("recall score", recall_score(y_test,y_pred))
print("f1 score", f1_score(y_pred,y_test))
