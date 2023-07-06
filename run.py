import os
import pandas as pd
from sklearn.metrics import log_loss
from config import Config
from feature import *
from model import *
from runner import Runner

Config().make_dir()

# 特徴量を作成・保存 ----------------------
feat_cls = [Example]

for feat_cl in feat_cls:
    if os.path.isfile(f'./feature/{feat_cl.__name__}_train.pkl'): # すでに作成した特徴量はskip
        print(f'[{feat_cl.__name__}] is already exist.')
        continue
    feat = feat_cl()
    feat.run().save()
    feat.create_memo()

# 特徴量 ----------------------------------
features = ['feat_1', 'feat_2', 'feat_new']

# 目的変数 --------------------------------
target = 'class'

# 実行名 ----------------------------------
run_name = 'run001'

# パラメータ ------------------------------
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
}

# 学習 ------------------------------------
# モデル、評価関数を適宜変更してください。
runner = Runner(run_name, ModelLGBM, features, target, params, log_loss)
runner.run_train_cv()
preds = runner.run_predict_cv()

# submission.csvの作成 -----------------------------------------------------
# 提出形式に応じて書き換えてください。
ids = pd.read_pickle(Config.test_path)['id']
sub = pd.DataFrame({'id': ids, 'prob': preds})
sub.to_csv(f'./submission/submission_{run_name}.csv', index=False, header=False)