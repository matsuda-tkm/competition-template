import pandas as pd
import os
import csv
import pickle
from config import Config

# 特徴量定義の基底クラス
class Feature:
    """docstringに特徴量の説明を記述"""
    prefix = ''
    suffix = ''

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.read_pickle(Config.train_path)
        self.test = pd.read_pickle(Config.test_path)

    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        return self

    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_pickle(f'./feature/{self.name}_train.pkl')
        self.test.to_pickle(f'./feature/{self.name}_test.pkl')

    def create_memo(self):
        file_path = './feature/_feature_memo.csv'
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as f:pass
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            col = [line for line in lines if line.split(',')[0] == self.name]
            if len(col) != 0:
                return
            writer = csv.writer(f)
            writer.writerow([self.name, self.__doc__])

# モデル定義の基底クラス
class ModelBase:
    def __init__(self, run_name, params=None):
        self.run_name = run_name
        self.params = params
        self.model = None
        self.y_pred = None

    def train(self, tr_x, tr_y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def save_model(self):
        pickle.dump(self.model, open(f'./model/{self.run_name}.pkl', 'wb'))

    def load_model(self):
        self.model = pickle.load(open(f'./model/{self.run_name}.pkl', 'rb'))
