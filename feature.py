from base import Feature

# 特徴量を定義 ------------------------------
class Example(Feature):
    """例"""
    def __init__(self):
        super().__init__()

    def create_features(self):
        self.train['feat_new'] = self.train['feat_1'] * self.train['feat_2']
        self.test['feat_new'] = self.test['feat_1'] * self.test['feat_2']
        self.feat_train = self.train[['feat_new']]
        self.feat_test = self.test[['feat_new']]