import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import Config

class Runner:
    def __init__(self, run_name, model_cls, features, target, params, metric, n_fold=4):
        """
        run_name  : str, 実行名
        model_cls : class, モデルクラス
        features  : list, 特徴量のリスト(ex. ['feat1', 'feat2', ...])
        target    : str, 目的変数
        params    : dict, モデルのハイパーパラメータ
        metric    : func, モデルの評価指標
        n_fold    : int, foldの個数
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.target = target
        self.params = params
        self.metric = metric
        self.n_fold = n_fold
        self.STDOUT = sys.stdout

        self.logger(f'features: {self.features}')
        self.logger(f'params: {self.params}')

    # 1fold分の学習
    def train_fold(self, i_fold):
        validation = (i_fold != 'all')
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if validation:
            # バリデーションデータの分割
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # モデルの学習
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y)

            # バリデーションデータに対する予測と評価
            va_pred = model.predict(va_x)
            score = self.metric(va_y, va_pred)

            return model, va_idx, va_pred, score
        else:
            # モデルの学習
            model = self.build_model(i_fold)
            model.train(train_x, train_y)

            return model, None, None, None

    # クロスバリデーションでの学習
    def run_train_cv(self):
        va_idxes = []
        va_preds = []
        scores = []

        for i_fold in range(self.n_fold):
            # 学習
            print(f'{self.run_name} - Fold {i_fold + 1}')
            with open(f'./model/{self.run_name}_calc.log', 'a') as f:
                sys.stdout = f
                print(f'\n---------- Fold {i_fold + 1} ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) ----------\n')
                model, va_idx, va_pred, score = self.train_fold(i_fold)
            sys.stdout = self.STDOUT
            # モデルの保存
            model.save_model()
            # 結果を保持
            va_idxes.append(va_idx)
            va_preds.append(va_pred)
            scores.append(score)
            # ログ出力
            print(f'Fold {i_fold} Score: {score}')
            self.logger(f'Fold {i_fold} Score: {score}')

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        va_preds = np.concatenate(va_preds, axis=0)[np.argsort(va_idxes)]

        # CVスコア(平均値)の表示
        print(f'{self.run_name} - training cv - score {np.mean(scores)}')
        self.logger(f'{self.run_name} - training cv - score {np.mean(scores)}')

    # 各foldのモデルをアンサンブルして予測
    def run_predict_cv(self):
        test_x = self.load_x_test()
        preds = []

        for i_fold in range(self.n_fold):
            # 学習済みモデルの読み込み
            model = self.build_model(i_fold)
            model.load_model()
            # 予測
            pred = model.predict(test_x)
            preds.append(pred)

        # 各foldの結果の平均をとる
        preds = np.mean(preds, axis=0)

        return preds

    # 全学習データで学習
    def run_train_all(self):
        model, _, _, _ = self.train_fold('all')
        model.save_model()

    # run_train_allで学習したモデルで予測
    def run_predict_all(self):
        test_x = self.load_x_test()

        model = self.build_model('all')
        model.load_model()
        preds = model.predict(test_x)

        return preds

    # モデルを作成(インスタンス化)
    def build_model(self, i_fold):
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    # foldを指定して対応するindexを返す
    def load_index_fold(self, i_fold):
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
        return list(skf.split(dummy_x, train_y))[i_fold]

    # 学習データの特徴量を読み込み
    def load_x_train(self):
        x_train = pd.read_pickle(Config.train_path)
        feat_origin = [feat for feat in self.features if feat in x_train.columns]
        feat_generated = [feat for feat in self.features if feat not in x_train.columns]
        if len(feat_generated) > 0:
            train_generated = pd.concat([pd.read_pickle(f'./feature/{feat}_train.pkl') for feat in feat_generated], axis=1)
            x_train = pd.concat([x_train[feat_origin], train_generated], axis=1)
        else:
            x_train = x_train[feat_origin]
        return x_train

    # 学習データの目的変数を読み込み
    def load_y_train(self):
        return pd.read_pickle(Config.train_path)[self.target]

    # テストデータの特徴量を読み込み
    def load_x_test(self):
        x_test = pd.read_pickle(Config.test_path)
        feat_origin = [feat for feat in self.features if feat in x_test.columns]
        feat_generated = [feat for feat in self.features if feat not in x_test.columns]
        if len(feat_generated) > 0:
            test_generated = pd.concat([pd.read_pickle(f'./feature/{feat}_test.pkl') for feat in feat_generated], axis=1)
            x_test = pd.concat([x_test[feat_origin], test_generated], axis=1)
        else:
            x_test = x_test[feat_origin]
        return x_test

    def logger(self, text):
        with open(f'./model/{self.run_name}_calc.log', 'a') as f:
            sys.stdout = f
            print(text)
        sys.stdout = self.STDOUT