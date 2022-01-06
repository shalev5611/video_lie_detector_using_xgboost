import xgboost
import os
import pandas as pd
import numpy as np
import subprocess
import docker

model = xgboost.XGBClassifier()
model.load_model("lies.json")

rows = ['frame', ' face_id', ' timestamp', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
au = [' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']
ex_au = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']


def most_frequent(List):
    return max(set(List), key = List.count)

def get_csv_from_docker(path, name):
    command = """sudo python3 -m http.server 8000 & nc -l -p 1234 > {}.csv & sudo docker run -it algebr/openface:latest -c "wget http://172.17.0.1:8000/{}.mp4; build/bin/FeatureExtraction -aus -f {}.mp4;ls processed; ifconfig; nc -w 3 172.17.0.1 1234 < processed/{}.csv";"""
    command = command.format(name,path,name,name)
    print(command)
    os.system(command)

def prepare_directory(path):
    X = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".csv") and entry.is_file():
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
                    en.append(most_frequent(row[i]))

                X.append(en)

    X = np.array(X)
    return X
def prepare_file(path):
    en = []
    csv = pd.read_csv(path)

    row = {}
    for i in rows:
        row[i] = []
        for j in csv[i]:
            row[i].append(j)

    for i in au:
        en.append(most_frequent(row[i]))
    for i in ex_au:
        en.append(most_frequent(row[i]))
    print(en)
    return np.array(en)

def predict(model, path,name):
    get_csv_from_docker(path,name)
    data = prepare_file(name + ".csv")
    print(data.shape)
    pred = model.predict(np.array([data]))
    print(pred)
