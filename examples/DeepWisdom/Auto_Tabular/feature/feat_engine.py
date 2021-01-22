from .feat_gen import *
from sklearn.utils import shuffle
from Auto_Tabular.utils.log_utils import log ,timeit

class FeatEngine:
    def __init__(self):

        #self.order2s = [GroupbyMeanMinusSelf]

        self.order2s = []

    def fit(self, data_space, order):
        if order != 2:
            return
        order_name = 'order{}s'.format(order)
        pipline = getattr(self, order_name)
        self.feats = []
        for feat_cls in pipline:
            feat = feat_cls()
            feat.fit(data_space)
            self.feats.append(feat)

    def transform(self, data_space, order):
        for feat in self.feats:
            feat.transform(data_space)

    @timeit
    def fit_transform(self, data_space, order, info=None):
        if order != 2:
            return
        order_name = 'order{}s'.format(order)
        pipline = getattr(self, order_name)
        X, y = data_space.data, data_space.y
        cats = data_space.cat_cols
        for feat_cls in pipline:
            feat = feat_cls()
            feat.fit_transform(X, y, cat_cols=cats, num_cols=info['imp_nums'])
        data_space.data = X
        data_space.update = True


        # all_index, train_idxs = data_space.all_idxs, data_space.all_train_idxs
        # test_idxs, cats = data_space.test_idxs, data_space.cat_cols
        # train_x, train_y = X.loc[train_idxs], y[train_idxs]
        # train_x, train_y = shuffle(train_x, train_y)
        # test_x = X.loc[test_idxs]
        # for feat_cls in pipline:
        #     feat = feat_cls(cols=cats)
        #     feat.fit_transform(train_x, train_y)
        #     feat.transform(test_x)
        # X = pd.concat([train_x, test_x])
        # X.sort_index(inplace=True)
        # data_space.data = X
