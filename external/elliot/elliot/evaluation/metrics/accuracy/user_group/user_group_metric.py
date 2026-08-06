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


def make_tail_metric(base_cls):
    """
    Return a new metric class that:
    - computes the same formula as base_cls
    - restricts evaluation to tail items that have less than maxitemcnt interactions in the training set
    - restricts evaluation to users that interacted with at least one tail item in the test set
    - counts users with no recommendations as 0
    """
 
    class _TailMetric(base_cls):
        def __init__(self, recommendations, config, params, eval_objects):
            # train_dict uses the same string user IDs as recommendations keys
            train_dict = eval_objects.data.train_dict
 
            # Compute item interaction counts in the training set
            item_counts = {}
            for items in train_dict.values():
                for item in items:
                    item_counts[item] = item_counts.get(item, 0) + 1
 
            # Identify tail items based on tail_ratio in config
            tail_ratio = getattr(config.evaluation, "long_tail_threshold", 0.1)
            print(f"tail_ratio: {tail_ratio}")
            if tail_ratio < 1:
                # Sort items by interaction count and keep the bottom tail_ratio fraction
                ncut = int(len(item_counts) * tail_ratio)
                tail_items = set(item for item, count in sorted(item_counts.items(), key=lambda x: x[1])[:ncut])
                print(f"Identified {len(tail_items)} tail items out of {len(item_counts)} total items.")
            else:
                # Identify tail items based on interaction count in training set
                maxitemcnt = int(tail_ratio)
                tail_items = {item for item, count in item_counts.items() if count < maxitemcnt}
 
            # Identify users who interacted with at least one tail item in the test set
            test_dict = eval_objects.data.test_dict
            tail_users = {u for u, items in test_dict.items() if any(i in tail_items for i in items)}
 
            # output recommendations only for tail users and only for tail items
            filtered = {
                u: [(item_id, score) for item_id, score in recommendations.get(u, []) if item_id in tail_items]
                for u in tail_users
            }
 
            super().__init__(filtered, config, params, eval_objects)
 
        @staticmethod
        def name():
            return f"{base_cls.name()}_tail"
 
    _TailMetric.__name__ = f"{base_cls.__name__}_tail"
    _TailMetric.__qualname__ = f"{base_cls.__qualname__}_tail"
    return _TailMetric
 