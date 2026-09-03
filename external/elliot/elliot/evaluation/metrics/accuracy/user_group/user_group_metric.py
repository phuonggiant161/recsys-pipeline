"""
Factory functions for user-group and item-popularity-group metric variants.

make_u1_metric(BaseCls)
    Restricts evaluation to users with ≤1 training interaction.

make_tail_metric(BaseCls)
    Restricts evaluation to tail items (defined by long_tail_threshold in config).
    NOTE: filters the recommendation list by tail items, which changes item ranks.

make_pop_recall_metric(group_name)
make_pop_ndcg_metric(group_name)
    Popularity-group Recall/nDCG with fixed bins:
        pop1 (count=1), pop2_5 (2-5), pop6_10 (6-10), pop11_20 (11-20), pop21_40 (21-40), pop41plus (>=41)
    Semantics:
    - Popularity = number of interactions in the TRAIN set.
    - Per user: denominator = |GT items in group|.
    - Users with no GT item in the group are excluded from the average.
    - Recommendation list is NEVER filtered — original rank positions are preserved.
    - DCG/IDCG use the position in the original top-K list, not a re-ranked subset.
    - GT is read from eval_objects.relevance.binary_relevance (correct for val/test split).
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared by pop-group metrics
# ──────────────────────────────────────────────────────────────────────────────

_POP_BINS = [
    ("pop1",      lambda c: c == 1),
    ("pop2_5",    lambda c: 2 <= c <= 5),
    ("pop6_10",   lambda c: 6 <= c <= 10),
    ("pop11_20",  lambda c: 11 <= c <= 20),
    ("pop21_40",  lambda c: 21 <= c <= 40),
    ("pop41plus", lambda c: c >= 41),
]


def _build_group_items(train_dict, group_name):
    """Return frozenset of item IDs whose train count falls in group_name."""
    item_counts = {}
    for items in train_dict.values():
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
    pred = dict(_POP_BINS)[group_name]
    return frozenset(item for item, cnt in item_counts.items() if pred(cnt))


# ──────────────────────────────────────────────────────────────────────────────
# Existing factories (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# User-activity group metrics
# Bins based on TRAIN interaction count per user:
#   u1 (==1), u2_5 (2-5), u6_10 (6-10), u11_20 (11-20), u21_40 (21-40), u41plus (>=41)
# Logic: filter USERS only. Full GT and full Top-K are preserved.
# ──────────────────────────────────────────────────────────────────────────────

_USER_BINS = {
    "u1":      lambda c: c == 1,
    "u2_5":    lambda c: 2 <= c <= 5,
    "u6_10":   lambda c: 6 <= c <= 10,
    "u11_20":  lambda c: 11 <= c <= 20,
    "u21_40":  lambda c: 21 <= c <= 40,
    "u41plus": lambda c: c >= 41,
}


def make_user_group_metric(base_cls, group_name):
    """
    Return a metric class restricted to users in the given train-activity group.

    User group is determined solely from TRAIN interaction count.
    After identifying group users:
    - Full recommendation list is passed (no item filtering, no re-ranking).
    - Full TEST GT is used (no item filtering).
    - Group users missing from recommendations get [] -> score = 0, still counted.
    - Group users in train but absent from current eval split: handled by base_cls.
    """
    pred = _USER_BINS[group_name]

    class _UserGroupMetric(base_cls):
        def __init__(self, recommendations, config, params, eval_objects):
            train_dict = eval_objects.data.train_dict

            group_users = {u for u, items in train_dict.items() if pred(len(items))}

            # Support statistics
            binary_rel  = eval_objects.relevance.binary_relevance
            eval_users  = group_users & set(binary_rel._binary_relevance.keys())
            n_interact  = sum(len(binary_rel.get_user_rel(u)) for u in eval_users)
            print(
                f"  [user_group_stats] {group_name}@cutoff={eval_objects.cutoff}: "
                f"train_users={len(group_users)}  "
                f"eval_users={len(eval_users)}  "
                f"eval_interactions={n_interact}"
            )

            # Full recs — no item filtering
            filtered = {u: recommendations.get(u, []) for u in group_users}

            super().__init__(filtered, config, params, eval_objects)

        def eval(self):
            vals = list(self.eval_user_metric().values())
            return float(np.mean(vals)) if vals else float("nan")

        @staticmethod
        def name():
            return f"{base_cls.name()}_{group_name}"

    cls_name = f"{base_cls.__name__}_{group_name}"
    _UserGroupMetric.__name__    = cls_name
    _UserGroupMetric.__qualname__ = cls_name
    return _UserGroupMetric


def make_u1_metric(base_cls):
    """
    Return a new metric class that:
    - computes the same formula as base_cls
    - restricts evaluation to users with train_interaction_count <= 1
    - counts users with no recommendations as 0
    """

    class _U1Metric(base_cls):
        def __init__(self, recommendations, config, params, eval_objects):
            train_dict = eval_objects.data.train_dict
            u1_users = {u for u, items in train_dict.items() if len(items) <= 1}
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
    - restricts evaluation to tail items (count < long_tail_threshold)
    - restricts evaluation to users that interacted with at least one tail item in test
    - filters recommendation list to tail items only (changes ranks)
    """

    class _TailMetric(base_cls):
        def __init__(self, recommendations, config, params, eval_objects):
            train_dict = eval_objects.data.train_dict

            item_counts = {}
            for items in train_dict.values():
                for item in items:
                    item_counts[item] = item_counts.get(item, 0) + 1

            tail_ratio = getattr(config.evaluation, "long_tail_threshold", 0.1)
            print(f"tail_ratio: {tail_ratio}")
            if tail_ratio < 1:
                ncut = int(len(item_counts) * tail_ratio)
                tail_items = set(
                    item for item, count in
                    sorted(item_counts.items(), key=lambda x: x[1])[:ncut]
                )
                print(f"Identified {len(tail_items)} tail items out of {len(item_counts)} total items.")
            else:
                maxitemcnt = int(tail_ratio)
                tail_items = {item for item, count in item_counts.items() if count < maxitemcnt}

            test_dict = eval_objects.data.test_dict
            tail_users = {u for u, items in test_dict.items() if any(i in tail_items for i in items)}

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


# ──────────────────────────────────────────────────────────────────────────────
# New: item-popularity-group metrics
# ──────────────────────────────────────────────────────────────────────────────

def make_pop_recall_metric(group_name):
    """
    Return a Recall class restricted to items in popularity group `group_name`.

    - denominator  = |GT items belonging to the group| per user
    - numerator    = hits in top-K using ORIGINAL rank positions (list never filtered)
    - eligible users: determined from GT (binary_relevance), NOT from recommendations
      A user qualifies iff they have ≥1 GT item in the group — regardless of
      whether their recommendation list contains any group item.
      Users missing from recommendations get an empty rec list → Recall = 0.
    - GT is sourced from eval_objects.relevance.binary_relevance so validation
      evaluation uses the validation split, not the test split
    - returns NaN when no eligible user exists in the group
    """
    from elliot.evaluation.metrics.base_metric import BaseMetric

    class _PopRecall(BaseMetric):
        def __init__(self, recommendations, config, params, eval_objects):
            train_dict  = eval_objects.data.train_dict
            binary_rel  = eval_objects.relevance.binary_relevance

            group_items = _build_group_items(train_dict, group_name)

            # Eligible users: iterate over ALL GT users (current split), not
            # over recommendations — so users with group GT but missing from
            # recommendations are still included (with empty rec list → score = 0)
            valid_users = {
                u for u in binary_rel._binary_relevance
                if any(i in group_items for i in binary_rel.get_user_rel(u))
            }

            # Support statistics
            n_test_interactions = sum(
                sum(1 for i in binary_rel.get_user_rel(u) if i in group_items)
                for u in valid_users
            )
            print(
                f"  [pop_stats] {group_name}@cutoff={eval_objects.cutoff}: "
                f"train_items={len(group_items)}  "
                f"test_users={len(valid_users)}  "
                f"test_interactions={n_test_interactions}"
            )

            # Pass FULL recommendations — do NOT filter by group (preserves ranks)
            filtered = {u: recommendations.get(u, []) for u in valid_users}

            self._group_items = group_items
            self._binary_rel  = binary_rel

            super().__init__(filtered, config, params, eval_objects)
            self._cutoff = eval_objects.cutoff

        @staticmethod
        def name():
            return f"Recall_{group_name}"

        def eval(self):
            vals = list(self.eval_user_metric().values())
            return float(np.mean(vals)) if vals else float("nan")

        def eval_user_metric(self):
            cutoff = self._cutoff
            result = {}
            for u, u_recs in self._recommendations.items():
                # GT items for user u in current split that belong to the group
                group_gt = [i for i in self._binary_rel.get_user_rel(u) if i in self._group_items]
                if not group_gt:
                    continue
                gt_set = set(group_gt)
                hits = sum(1 for i, _ in u_recs[:cutoff] if i in gt_set)
                result[u] = hits / len(group_gt)
            return result

    cls_name = f"Recall_{group_name}"
    _PopRecall.__name__ = cls_name
    _PopRecall.__qualname__ = cls_name
    return _PopRecall


def make_pop_ndcg_metric(group_name):
    """
    Return an nDCG class restricted to items in popularity group `group_name`.

    - IDCG is computed over min(|GT items in group|, cutoff) ideal positions
    - DCG uses the ORIGINAL rank position of each relevant item in the full top-K
      (list is NOT filtered — rank 17 stays rank 17)
    - eligible users: determined from GT (binary_relevance), NOT from recommendations
      Users missing from recommendations get an empty rec list → nDCG = 0.
    - GT is sourced from eval_objects.relevance.binary_relevance (correct split)
    - returns NaN when no eligible user exists in the group
    """
    from elliot.evaluation.metrics.base_metric import BaseMetric

    class _PopNDCG(BaseMetric):
        def __init__(self, recommendations, config, params, eval_objects):
            train_dict  = eval_objects.data.train_dict
            binary_rel  = eval_objects.relevance.binary_relevance

            group_items = _build_group_items(train_dict, group_name)

            # Same GT-based eligibility as _PopRecall
            valid_users = {
                u for u in binary_rel._binary_relevance
                if any(i in group_items for i in binary_rel.get_user_rel(u))
            }

            filtered = {u: recommendations.get(u, []) for u in valid_users}

            self._group_items = group_items
            self._binary_rel  = binary_rel

            super().__init__(filtered, config, params, eval_objects)
            self._cutoff = eval_objects.cutoff

        @staticmethod
        def name():
            return f"nDCG_{group_name}"

        def eval(self):
            vals = list(self.eval_user_metric().values())
            return float(np.mean(vals)) if vals else float("nan")

        def eval_user_metric(self):
            cutoff = self._cutoff
            result = {}
            for u, u_recs in self._recommendations.items():
                group_gt = frozenset(
                    i for i in self._binary_rel.get_user_rel(u) if i in self._group_items
                )
                if not group_gt:
                    continue
                # DCG: use original rank (r is 0-indexed position in full top-K)
                dcg = sum(
                    1.0 / np.log2(r + 2)
                    for r, (i, _) in enumerate(u_recs[:cutoff])
                    if i in group_gt
                )
                # IDCG: best case with len(group_gt) items at top positions
                n_rel = min(len(group_gt), cutoff)
                idcg = sum(1.0 / np.log2(r + 2) for r in range(n_rel))
                result[u] = dcg / idcg if idcg > 0 else float("nan")
            return result

    cls_name = f"nDCG_{group_name}"
    _PopNDCG.__name__ = cls_name
    _PopNDCG.__qualname__ = cls_name
    return _PopNDCG
