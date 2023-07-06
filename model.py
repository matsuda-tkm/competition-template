import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from base import ModelBase


# 使いたいモデルを定義 ------------------------
class ModelLGBM(ModelBase):
    """LightGBM"""
    def __init__(self, run_name ,params=None):
        super().__init__(run_name, params)

    def train(self, tr_x, tr_y):
        from sklearn.model_selection import train_test_split
        tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, train_size=0.8)
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train)
        self.model = lgb.train(self.params,
                               lgb_train,
                               valid_sets=lgb_eval,
                               )

    def predict(self, x):
        self.y_pred = self.model.predict(x)
        return self.y_pred

    def save_model(self):
        pickle.dump(self.model, open(f'./model/{self.run_name}.pkl', 'wb'))
        # 特徴量重要度のプロットも保存
        lgb.plot_importance(self.model)
        plt.savefig(f'./model/{self.run_name}_feature_importance.png')

class ModelLR(ModelBase):
    """ロジスティック回帰"""
    def __init__(self, run_name, params=None):
        super().__init__(run_name, params)
        self.model = LogisticRegression(**self.params)

    def train(self, tr_x, tr_y):
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        self.y_pred = self.model.predict_proba(x)[:, 1]
        return self.y_pred

class ModelRFC(ModelBase):
    """ランダムフォレスト"""
    def __init__(self, run_name, params=None):
        super().__init__(run_name, params)
        self.model = RandomForestClassifier(**self.params)

    def train(self, tr_x, tr_y):
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        self.y_pred = self.model.predict_proba(x)[:, 1]
        return self.y_pred