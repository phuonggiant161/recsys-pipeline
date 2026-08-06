"""
Factory for creating user-group-filtered variants of existing metrics.

A metric created via make_u1_metric(BaseCls) computes the same formula
as BaseCls but restricts evaluation to users whose training interaction
count is <= 1.  Users in the group who received no recommendations are
included with a metric value of 0 (not silently dropped).
"""


def make_u1_metric(base_cls):
    """
    Return a new metric class that:
    - computes the same formula as base_cls
    - restricts evaluation to users with train_interaction_count <= 1
    - counts users with no recommendations as 0
    """

    class _U1Metric(base_cls):
        def __init__(self, recommendations, config, params, eval_objects):
            # train_dict uses the same string user IDs as recommendations keys
            train_dict = eval_objects.data.train_dict

            u1_users = {u for u, items in train_dict.items() if len(items) <= 1}

            # Every u1 user gets an entry so users with no recs score 0
            filtered = {u: recommendations.get(u, []) for u in u1_users}

            super().__init__(filtered, config, params, eval_objects)

        @staticmethod
        def name():
            return f"{base_cls.name()}_u1"

    _U1Metric.__name__ = f"{base_cls.__name__}_u1"
    _U1Metric.__qualname__ = f"{base_cls.__qualname__}_u1"
    return _U1Metric
