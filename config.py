import os
import pandas as pd
import pickle

class Config:
    # train,testのパスを指定 ---------------
    train_path_csv = './data/train.csv'
    test_path_csv = './data/test.csv'
    # --------------------------------------
    train_path = train_path_csv.replace('.csv', '.pkl')
    test_path = test_path_csv.replace('.csv', '.pkl')

    def __init__(self):
        pass

    def make_dir(self):
        # ディレクトリ作成
        os.makedirs('./feature', exist_ok=True)
        os.makedirs('./model', exist_ok=True)
        os.makedirs('./submission', exist_ok=True)
        # csvファイル読み込み
        train = pd.read_csv(self.train_path_csv)
        test = pd.read_csv(self.test_path_csv)
        # 前処理
        train, test = self.preprocess(train, test)
        # pickleファイル保存
        pickle.dump(train, open(self.train_path, 'wb'))
        pickle.dump(test, open(self.test_path, 'wb'))

    def preprocess(self, train, test):
        # 前処理をここに書く -----------

        # ------------------------------
        return train, test