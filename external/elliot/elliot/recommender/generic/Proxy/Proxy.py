import ntpath
import numpy as np
import pandas as pd

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class ProxyRecommender(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Proxy recommender to evaluate already generated recommendations.
        :param name: data loader object
        :param path: path to the directory rec. results
        :param args: parameters
        """
        self._random = np.random

        self._params_list = [
            ("_name", "name", "name", "", None, None),
            ("_path", "path", "path", "", None, None)
        ]
        self.autoset_params()
        if not self._name:
            self._name = ntpath.basename(self._path).rsplit(".",1)[0]

    @property
    def name(self):
        return self._name

    def train(self):
        print("Reading recommendations")
        self._recommendations = self.read_recommendations(self._path)

        print("Evaluating recommendations")
        self.evaluate()

    def get_recommendations(self, top_k):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(top_k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k):
        # Avoid materializing the full dense candidate mask, which can OOM on
        # large datasets. For the current full-ranking evaluation protocol, the
        # candidate set is catalog items not seen in TRAIN, so filtering saved
        # recommendations against each user's sparse TRAIN item set is equivalent
        # to the previous candidate-mask membership check.
        cx = self._data.sp_i_train.tocsr()
        train_items_per_user = {
            self._data.private_users[u]: {
                self._data.private_items[i] for i in cx.getrow(u).indices
            }
            for u in range(cx.shape[0])
        }

        recs = {}
        for u, user_recs in self._recommendations.items():
            train_set = train_items_per_user.get(u, set())
            user_cleaned_recs = []
            for p, (item, prediction) in enumerate(user_recs):
                if p >= k:
                    break
                if item not in train_set:
                    user_cleaned_recs.append((item, prediction))
            recs[u] = user_cleaned_recs
        return recs

    def read_recommendations(self, path):
        recs = {}
        column_names = ["userId", "itemId", "prediction"]
        data = pd.read_csv(path, sep="\t", header=None, names=column_names, usecols=[0, 1, 2])
        user_groups = data.groupby('userId')
        for name, group in user_groups:
            recs[name] = sorted(group[['itemId', 'prediction']].apply(tuple, axis=1).to_list(), key=lambda x: x[1], reverse=True)
        return recs



